#!/usr/bin/env python3
"""Independently audit saved I2CL runs, labels, support isolation and metrics.

This CPU-only audit recomputes metrics and artifact identities. Runtime tensor
freeze, gradient and checkpoint-reload records are cross-checked against the
saved checkpoint/history; it does not rerun the CUDA forward or backward pass.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import torch

from batch_data import load_dataset
from i2cl_data import load_spec
from infer import ASSETS_DIR, MODEL_DIR
from infer_batch import checkpoint_identity


UNWEIGHTED_LOSS = "Unweighted BCEWithLogitsLoss over every original binary target"
SUPPORT_WEIGHTED_LOSS = "BCEWithLogitsLoss with per-class support negatives/positives pos_weight"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(spec):
    return {
        "image_paths": list(spec["image_paths"]),
        "labels": np.asarray(spec["labels"], dtype=np.int8).tolist(),
        "ids": list(spec["ids"]),
        "groups": list(spec["groups"]),
        "class_names": list(spec["class_names"]),
        "prompts": list(spec["prompts"]),
    }


def close(actual, expected, description):
    if expected is None:
        require(actual is None, f"{description}: expected null, got {actual}")
    else:
        require(actual is not None and math.isfinite(float(actual)) and
                math.isclose(float(actual), float(expected), abs_tol=1e-11, rel_tol=1e-10),
                f"{description}: {actual} != {expected}")


def independent_metrics(labels, logits, classes):
    def binary(target, score):
        positive = int(np.sum(target))
        negative = len(target) - positive
        roc = float(roc_auc_score(target, score)) if positive and negative else None
        if positive:
            precision, recall, _ = precision_recall_curve(target, score)
            pr = float(auc(recall, precision))
        else:
            pr = None
        return {"positives": positive, "negatives": negative, "auroc": roc, "auprc": pr}

    def average(values):
        defined = [value for value in values if value is not None]
        return float(np.mean(defined)) if defined else None

    per_class = {name: binary(labels[:, index], logits[:, index])
                 for index, name in enumerate(classes)}
    micro = binary(labels.ravel(), logits.ravel())
    return {
        "sample_count": len(labels),
        "class_count": len(classes),
        "per_class": per_class,
        "macro_auroc": average([item["auroc"] for item in per_class.values()]),
        "macro_auprc": average([item["auprc"] for item in per_class.values()]),
        "micro_auroc": micro["auroc"],
        "micro_auprc": micro["auprc"],
        "macro_auroc_defined_classes": sum(item["auroc"] is not None for item in per_class.values()),
        "macro_auprc_defined_classes": sum(item["auprc"] is not None for item in per_class.values()),
    }


def verify_metrics(saved, measured, description):
    for key in ("sample_count", "class_count", "macro_auroc_defined_classes",
                "macro_auprc_defined_classes"):
        require(saved[key] == measured[key], f"{description}: {key} mismatch")
    for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc"):
        close(saved[key], measured[key], f"{description}: {key}")
    require(set(saved["per_class"]) == set(measured["per_class"]),
            f"{description}: different classes")
    for name, values in measured["per_class"].items():
        for key in ("positives", "negatives"):
            require(saved["per_class"][name][key] == values[key],
                    f"{description}/{name}: {key} mismatch")
        for key in ("auroc", "auprc"):
            close(saved["per_class"][name][key], values[key], f"{description}/{name}/{key}")


def verify_loss_weighting(config, training, support):
    """Recompute new loss weights from supports without changing historical audits."""
    if config["loss"] != SUPPORT_WEIGHTED_LOSS:
        return None
    require(config["class_names"] == support["class_names"],
            "Loss-weight class order differs from the support labels")
    labels = np.asarray(support["labels"])
    require(labels.ndim == 2 and labels.shape[1] == len(config["class_names"]) and
            np.isin(labels, [0, 1]).all(), "Loss weights require binary support targets")
    positives = labels.sum(axis=0, dtype=np.int64)
    negatives = len(labels) - positives
    require(np.all(positives > 0), "Cannot compute pos_weight for a class without support positives")
    # Match the float32 target-tensor division in training, including saved rounding.
    weights = negatives.astype(np.float32) / positives.astype(np.float32)
    expected = {
        "statistics_split": "support",
        "positive_counts": positives.tolist(),
        "negative_counts": negatives.tolist(),
        "pos_weight": weights.tolist(),
        "reduction": "mean",
    }
    require(config.get("loss_weighting") == expected,
            "Configured pos_weight does not match support-only negative/positive counts")
    require(training.get("loss_weighting") == expected,
            "Training loss weights differ from the configured support-only weights")
    return expected


def verify_regularization(config, training, history):
    """Audit the additive identity penalty while accepting historical BCE-only runs."""
    fields = (
        "initial_coefficient_penalty", "final_coefficient_penalty",
        "initial_regularization_loss", "final_regularization_loss",
        "initial_support_total_loss_without_noise", "final_support_total_loss_without_noise",
    )
    history_fields = (
        "train_coefficient_penalty", "train_regularization_loss", "train_total_loss_with_noise",
    )
    if "regularization" not in config:
        require("lambda_reg" not in config["arguments"] and
                "regularization" not in training and "objective" not in config and
                not any(key in training for key in fields) and
                not any(key in epoch for epoch in history for key in history_fields),
                "Regularization records are incomplete")
        return None
    strength = config["arguments"].get("lambda_reg")
    require(isinstance(strength, (int, float)) and not isinstance(strength, bool) and
            math.isfinite(strength) and strength >= 0,
            "Regularization lambda must be finite and nonnegative")
    expected = {
        "type": "identity_coefficients_l2",
        "lambda": strength,
        "formula": "mean(alpha^2) + mean((beta - 1)^2)",
        "alpha_target": 0.0,
        "beta_target": 1.0,
    }
    require(config["regularization"] == training.get("regularization") == expected,
            "Configured and saved coefficient regularization differ")
    require(config.get("objective") ==
            "weighted BCE + lambda_reg * (mean(alpha^2) + mean((beta - 1)^2))" and
            config["loss"] == SUPPORT_WEIGHTED_LOSS,
            "Unexpected regularized objective or classification loss")

    def check_value(actual, target, description):
        # Runtime loss arithmetic uses float32 on the GPU; the independent audit
        # uses float64 and must allow reduction/multiplication rounding.
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                actual >= 0 and math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-7),
                f"{description}: {actual} != {target}")

    for phase in ("initial", "final"):
        values = np.asarray(training[f"{phase}_coefficients"], dtype=np.float64)
        require(values.shape == (2, 2, 2) and np.isfinite(values).all(),
                f"Invalid {phase} coefficients for regularization audit")
        penalty = float(np.mean(values[..., 0] ** 2) +
                        np.mean((values[..., 1] - 1) ** 2))
        check_value(training[f"{phase}_coefficient_penalty"], penalty,
                    f"{phase} coefficient penalty")
        weighted = strength * penalty
        check_value(training[f"{phase}_regularization_loss"], weighted,
                    f"{phase} weighted coefficient penalty")
        bce = training[f"{phase}_support_bce_without_noise"]
        require(isinstance(bce, (int, float)) and math.isfinite(bce) and bce >= 0,
                f"Invalid {phase} support BCE")
        check_value(training[f"{phase}_support_total_loss_without_noise"], bce + weighted,
                    f"{phase} total support objective")
    for epoch in history:
        penalty = epoch["train_coefficient_penalty"]
        require(isinstance(penalty, (int, float)) and math.isfinite(penalty) and penalty >= 0,
                "Invalid epoch mean coefficient penalty")
        weighted = strength * penalty
        check_value(epoch["train_regularization_loss"], weighted,
                    "Epoch mean weighted coefficient penalty")
        check_value(epoch["train_total_loss_with_noise"],
                    epoch["train_bce_with_noise"] + weighted,
                    "Epoch mean total objective")
    return expected


def read_predictions(path, spec):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(spec["image_paths"]), f"{path}: incomplete predictions")
    logits = []
    scores = []
    for index, (row, image_path, target) in enumerate(
            zip(rows, spec["image_paths"], spec["labels"])):
        require(int(row["index"]) == index, f"{path}: index order changed")
        require(row["image_path"] == image_path, f"{path}: image order changed at {index}")
        actual_target = [int(row[f"label:{name}"]) for name in spec["class_names"]]
        require(actual_target == np.asarray(target).tolist(), f"{path}: label mismatch at {index}")
        logits.append([float(row[f"logit:{name}"]) for name in spec["class_names"]])
        scores.append([float(row[f"score:{name}"]) for name in spec["class_names"]])
    logits = np.asarray(logits, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float64)
    require(np.isfinite(logits).all() and np.isfinite(scores).all(),
            f"{path}: nonfinite prediction")
    expected_scores = np.empty_like(scores)
    positive = logits >= 0
    expected_scores[positive] = 1 / (1 + np.exp(-logits[positive].astype(np.float64)))
    exp_value = np.exp(logits[~positive].astype(np.float64))
    expected_scores[~positive] = exp_value / (1 + exp_value)
    require(np.allclose(scores, expected_scores, rtol=1e-12, atol=1e-14),
            f"{path}: sigmoid scores disagree with logits")
    return logits


def verify_source_targets(name, support, full_test):
    original = load_dataset(name, ASSETS_DIR / "data")
    labels = dict(zip(original["image_paths"], original["labels"].tolist()))
    for path, target in zip(full_test["image_paths"], full_test["labels"]):
        require(labels.get(path) == np.asarray(target).tolist(), f"{name}: test source label mismatch")
    if name == "SIIM" and support["metadata"]["protocol"] == "released_test_fewshot_group_holdout":
        verify_siim_holdout_groups(original, support, full_test)
    elif name == "SIIM":
        clean_path = ASSETS_DIR / "splits/siim_adapter_v1/train.csv"
        with clean_path.open(newline="", encoding="utf-8") as handle:
            labels = {row["image_path"]: [int(row["label"])] for row in csv.DictReader(handle)}
        require(full_test["image_paths"] == original["image_paths"],
                "SIIM original test order/count changed")
    for path, target in zip(support["image_paths"], support["labels"]):
        require(labels.get(path) == np.asarray(target).tolist(), f"{name}: support source label mismatch")
    return original


def verify_support_subset(split_dir, name, manifest, support, test):
    """Audit the reduced support experiment without releasing old holdout groups."""
    if manifest["protocol"] != "support_subset_fixed_test":
        return None
    require(name in ("ChestXDet10", "ChestXray14", "OpenI"),
            "Fixed-test support subset is only defined for the three multilabel datasets")
    require(manifest["shots_per_class"] == 1, "Expected one anchor image per disease")
    parent = manifest["parent_split"]
    parent_root = Path(parent["path"]).resolve()
    parent_directory = parent_root / name
    require(parent_root != Path(split_dir).resolve(), "Subset cannot be its own parent")
    require(Path(parent["manifest_path"]).resolve() == parent_directory / "manifest.json",
            "Parent manifest does not belong to the declared dataset")
    for filename, field in (
        ("manifest.json", "manifest_sha256"), ("support.csv", "support_sha256"),
        ("test.csv", "test_sha256"), ("inventory.csv", "inventory_sha256"),
        ("excluded.csv", "excluded_sha256"),
    ):
        require(sha256(parent_directory / filename) == parent[field],
                f"Parent split changed: {name}/{filename}")
    require(sha256(parent_root / "manifest.json") == parent["root_manifest_sha256"],
            "Parent root manifest changed")
    parent_manifest = read_json(parent_directory / "manifest.json")
    require(parent_manifest["fingerprint"] == parent["fingerprint"],
            "Parent dataset fingerprint differs")
    require(parent_manifest["shots_per_class"] == 5 and
            parent_manifest["protocol"] == "released_test_fewshot_group_holdout",
            "Subset parent is not the previous five-shot holdout")
    parent_support = load_spec(parent_root, name, "support")
    parent_test = load_spec(parent_root, name, "test")
    directory = Path(split_dir) / name
    for filename in ("test.csv", "inventory.csv"):
        require(sha256(directory / filename) == sha256(parent_directory / filename) ==
                manifest["csv_sha256"][filename], f"{filename} differs from the fixed parent")
    require(identity(test) == identity(parent_test),
            "Fixed test image order, labels, classes or groups changed")
    for key in ("class_names", "prompts", "data_root", "seed", "grouping"):
        require(manifest[key] == parent_manifest[key], f"Subset changed parent {key}")

    def rows(path):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def without_index(row):
        return {key: value for key, value in row.items() if key != "index"}

    parent_rows = rows(parent_directory / "support.csv")
    current_rows = rows(directory / "support.csv")
    parent_lookup = {row["image_path"]: index for index, row in enumerate(parent_rows)}
    current_paths = [row["image_path"] for row in current_rows]
    require(len(set(current_paths)) == len(current_paths) < len(parent_rows) and
            set(current_paths) < set(parent_lookup), "Support must be a strict unique parent subset")
    indices = [parent_lookup[path] for path in current_paths]
    require(indices == sorted(indices), "Subset changed the original support order")
    require(all(without_index(row) == without_index(parent_rows[index])
                for row, index in zip(current_rows, indices)),
            "Subset changed source labels, anchors or patient/study/hash identities")
    selection = manifest["selection"]
    require(selection["method"] == "one_per_parent_anchor_class" and
            selection["uses_test_labels"] is False and
            selection["seed"] == manifest["seed"] and
            selection["parent_support_indices"] == indices and
            selection["eligible_support_candidates"] == len(parent_rows),
            "Subset selection provenance differs")
    require(selection["anchor_order"] == parent_manifest["selection"]["anchor_order"],
            "Subset changed the parent anchor order")
    anchors = [row["anchor_class"] for row in current_rows]
    require(len(current_rows) == len(support["class_names"]) and
            set(anchors) == set(support["class_names"]) and
            all(anchors.count(name) == 1 for name in support["class_names"]) and
            selection["anchor_counts"] == {name: 1 for name in support["class_names"]},
            "Subset does not contain exactly one anchor per class")
    labels = np.asarray(support["labels"])
    require(np.all(labels.sum(axis=0) > 0) and np.all(labels.sum(axis=0) < len(labels)),
            "Subset needs positive and negative targets for every class")
    unused = [row["image_path"] for row in parent_rows if row["image_path"] not in current_paths]
    require(manifest["unused_parent_support_image_paths"] == unused and
            manifest["unused_parent_support_images"] == len(unused),
            "Unused parent support accounting differs")
    require(not (set(unused) & set(test["image_paths"])),
            "Unused parent support was returned to the test set")
    parent_excluded = rows(parent_directory / "excluded.csv")
    require(manifest["parent_excluded_images"] == len(parent_excluded),
            "Parent excluded image count differs")
    expected_excluded = {row["image_path"]: row for row in parent_excluded}
    expected_excluded.update({row["image_path"]: row for row in parent_rows
                              if row["image_path"] in unused})
    inventory = rows(parent_directory / "inventory.csv")
    expected_excluded_order = [row["image_path"] for row in inventory
                               if row["image_path"] in expected_excluded]
    excluded = rows(directory / "excluded.csv")
    require(sha256(directory / "excluded.csv") == manifest["csv_sha256"]["excluded.csv"] and
            [row["image_path"] for row in excluded] == expected_excluded_order,
            "Excluded images differ from parent excluded plus unused support")
    require(all(without_index(row) == without_index(expected_excluded[row["image_path"]])
                for row in excluded), "Excluded row metadata differs")
    for key in ("image_paths", "ids", "groups"):
        require(not set(parent_support[key]) & set(test[key]),
                f"An original support-associated {key} entered the fixed test")
    for key in ("patient_ids", "study_ids", "byte_sha256"):
        require(not ({value for value in parent_support["metadata"][key] if value} &
                     {value for value in test["metadata"][key] if value}),
                f"An original support-associated {key} entered the fixed test")
    return {
        "status": "passed", "parent_split_root": str(parent_root),
        "parent_support_images": len(parent_rows), "support_images": len(current_rows),
        "unused_parent_support_images": len(unused),
        "parent_support_sha256": parent["support_sha256"],
        "fixed_test_sha256": parent["test_sha256"],
        "test_csv_byte_identical": True, "inventory_csv_byte_identical": True,
        "strict_support_subset": True, "one_anchor_per_class": True,
        "unused_support_returned_to_test": False,
    }


def verify_siim_holdout_groups(original, support, test):
    """Check the explicitly authorized test-pool holdout against actual DICOM IDs."""
    import pydicom

    manifest_path = Path(support["metadata"]["split_manifest"])
    manifest = read_json(manifest_path)
    inventory_path = manifest_path.parent / "inventory.csv"
    require(sha256(inventory_path) == manifest["csv_sha256"]["inventory.csv"],
            "SIIM inventory checksum mismatch")
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require([row["image_path"] for row in rows] == original["image_paths"],
            "SIIM inventory differs from the original test image order")
    require([json.loads(row["labels_json"]) for row in rows] == original["labels"].tolist(),
            "SIIM inventory labels differ from original test targets")
    root = Path(support["data_root"])
    identity_groups = {}
    for row in rows:
        require(row["source_split"] == "released_test", "SIIM holdout source differs")
        header = pydicom.dcmread(
            root / row["image_path"], stop_before_pixels=True,
            specific_tags=["PatientID", "StudyInstanceUID", "SOPInstanceUID"],
        )
        for column, tag in (("patient_id", "PatientID"), ("study_id", "StudyInstanceUID"),
                            ("image_id", "SOPInstanceUID")):
            value = str(getattr(header, tag, "")).strip()
            require(bool(value) and value == row[column],
                    f"SIIM inventory {column} disagrees with DICOM: {row['image_path']}")
        for column in ("patient_id", "study_id", "byte_sha256"):
            key = (column, row[column])
            previous = identity_groups.setdefault(key, row["group"])
            require(previous == row["group"], f"SIIM {column} crosses recorded groups")
    lookup = {row["image_path"]: row for row in rows}
    support_groups = set(support["groups"])
    expected_test = [row["image_path"] for row in rows if row["group"] not in support_groups]
    require(test["image_paths"] == expected_test,
            "SIIM evaluation is not the original order minus every support-associated group")
    for spec in (support, test):
        for index, path in enumerate(spec["image_paths"]):
            row = lookup[path]
            require(spec["groups"][index] == row["group"], "SIIM split/inventory group mismatch")
            for key, column in (("patient_ids", "patient_id"), ("study_ids", "study_id"),
                                ("byte_sha256", "byte_sha256")):
                require(spec["metadata"][key][index] == row[column],
                        f"SIIM split/inventory {column} mismatch")
    require(manifest["support_rows_in_released_evaluation"] == len(support["image_paths"]),
            "SIIM exception support count mismatch")
    require(manifest["removed_from_released_evaluation"] ==
            len(original["image_paths"]) - len(test["image_paths"]),
            "SIIM exception removed-image count mismatch")


def verify_baseline(config, baseline, spec, original, verification):
    saved = config["baseline"]
    require(verification["passed"], "Runtime baseline verification did not pass")
    if saved["mode"] == "computed_in_same_evaluation":
        require(verification["checked_images"] == len(spec["image_paths"]),
                "Recomputed baseline did not cover every held-out image")
        require(verification.get("method") ==
                "Base logits recomputed on every held-out image with injection disabled",
                "Unexpected same-evaluation baseline runtime record")
        require(baseline.shape == np.asarray(spec["labels"]).shape and
                np.isfinite(baseline).all(), "Invalid same-evaluation baseline predictions")
        return
    require(saved["mode"] == "original_predictions_reindexed_by_image_path_and_checked_labels",
            "Unknown baseline provenance mode")
    predictions_path = Path(saved["predictions_path"])
    manifest_path = Path(saved["manifest_path"])
    require(sha256(predictions_path) == saved["predictions_sha256"], "Original baseline CSV changed")
    require(sha256(manifest_path) == saved["manifest_sha256"], "Original baseline manifest changed")
    original_identity = {
        "paths": original["image_paths"],
        "classes": original["class_names"],
        "prompts": original["prompts"],
        "labels": original["labels"].tolist(),
        "model": str(MODEL_DIR),
        "checkpoint": checkpoint_identity(),
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "protocol": "official-test-preprocessing-fp32-v1",
    }
    expected = hashlib.sha256(json.dumps(original_identity, sort_keys=True).encode()).hexdigest()
    require(read_json(manifest_path)["fingerprint"] == saved["source_fingerprint"] == expected,
            "Original baseline fingerprint differs from current source/checkpoint/prompts")
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    lookup = {row["image_path"]: row for row in records}
    require(len(lookup) == len(records) and set(lookup) == set(original["image_paths"]),
            "Original baseline does not uniquely cover its complete source test")
    expected_logits = []
    for path, labels in zip(spec["image_paths"], spec["labels"]):
        row = lookup[path]
        require([int(row[f"label:{name}"]) for name in spec["class_names"]] == labels.tolist(),
                "Reindexed baseline labels differ")
        expected_logits.append([float(row[f"logit:{name}"]) for name in spec["class_names"]])
    require(np.array_equal(baseline, np.asarray(expected_logits, dtype=np.float32)),
            "Reindexed baseline logits differ from old public5 predictions")


def audit_dataset(directory, job, allow_smoke):
    name = directory.name
    config = read_json(directory / "run_config.json")
    result = read_json(directory / "result.json")
    args = config["arguments"]
    require(config["dataset"] == result["dataset"] == name, "Dataset identity mismatch")
    require(result["status"] == "complete", "Worker result is incomplete")
    require(not config["is_smoke_test"] or allow_smoke, "Smoke run requires --allow-smoke")
    require(config["is_smoke_test"] == result["is_smoke_test"] == bool(args["limit_test"]),
            "Smoke/full flags disagree")
    require(fingerprint({key: value for key, value in config.items() if key != "fingerprint"})
            == config["fingerprint"] == result["run_fingerprint"], "Run fingerprint mismatch")
    for mapping in ("source_sha256", "split_sha256"):
        for path, expected in config[mapping].items():
            require(sha256(path) == expected, f"Changed {mapping} artifact: {path}")
    require(config["checkpoint"] == checkpoint_identity(), "Official model weights differ")
    require(sha256(MODEL_DIR / "config.json") == config["model_config_sha256"],
            "Official model config changed")
    require(sha256(MODEL_DIR / "preprocessor_config.json") == config["preprocessor_sha256"],
            "Official preprocessor changed")
    require(config["resolution"] == result["resolution"] == [518, 518], "Wrong resolution")
    physical_gpu = str(job["gpu"])
    visible_gpu = config["cuda_visible_devices"]
    require(physical_gpu in tuple(str(index) for index in range(8)) and
            visible_gpu == result["physical_gpu"],
            "Physical GPU identity mismatch")
    if str(visible_gpu).startswith("GPU-"):
        require(job["gpu_info"]["physical_index"] == job["gpu"] and
                job["gpu_info"]["uuid"] == visible_gpu,
                "Visible GPU UUID does not match the recorded physical device")
    else:
        require(visible_gpu == physical_gpu, "Visible GPU index differs from physical device")
    split_dir = Path(args["split_dir"])
    support = load_spec(split_dir, name, "support")
    full_test = load_spec(split_dir, name, "test")
    spec = dict(full_test)
    limit = args["limit_test"]
    if limit and limit < len(full_test["image_paths"]):
        for key in ("image_paths", "labels", "ids", "groups"):
            spec[key] = full_test[key][:limit]
    support_count, test_count = len(support["image_paths"]), len(spec["image_paths"])
    require(config["support_count"] == result["support_count"] == support_count,
            "Support count mismatch")
    require(config["evaluated_count"] == result["evaluated_count"] == test_count,
            "Evaluation count mismatch")
    require(config["full_holdout_count"] == result["full_holdout_count"] ==
            len(full_test["image_paths"]), "Full holdout count mismatch")
    split_manifest = read_json(split_dir / name / "manifest.json")
    for item in split_manifest["source_files"]:
        require(sha256(item["path"]) == item["sha256"], f"Split source changed: {item['path']}")
    for key in ("image_paths", "ids", "groups"):
        require(not (set(support[key]) & set(full_test[key])), f"Support/test {key} overlap")
    for key in ("patient_ids", "study_ids", "byte_sha256"):
        require(not ({x for x in support["metadata"][key] if x} &
                     {x for x in full_test["metadata"][key] if x}), f"Support/test {key} overlap")
    anchors = support["metadata"]["anchor_classes"]
    expected_anchors = (support["class_names"] if len(support["class_names"]) > 1 else
                        [f"{support['class_names'][0]}:negative",
                         f"{support['class_names'][0]}:positive"])
    require(set(anchors) == set(expected_anchors), "Anchor labels differ")
    require(all(anchors.count(anchor) == split_manifest["shots_per_class"]
                for anchor in expected_anchors), "Anchor quota mismatch")
    for anchor, target in zip(anchors, support["labels"]):
        if len(support["class_names"]) > 1:
            require(target[support["class_names"].index(anchor)] == 1, "Nonpositive anchor assignment")
        else:
            require(target[0] == int(anchor.endswith(":positive")), "Wrong binary anchor state")
    original = verify_source_targets(name, support, full_test)
    subset_verification = verify_support_subset(
        split_dir, name, split_manifest, support, full_test
    )
    support_identity = fingerprint(identity(support))
    test_identity = fingerprint(identity(spec))
    require(test_identity == config["test_identity"], "Test identity mismatch")
    saved_identity = read_json(directory / "support_identity.json")
    require({key: saved_identity[key] for key in identity(support)} == identity(support),
            "Saved support images/targets differ")
    cache = read_json(directory / "support_cache.json")
    for record in (config, result, saved_identity, cache):
        require(all(record[key] == support_identity for key in
                    ("support_identity", "context_support_identity", "calibration_support_identity")),
                "Context and calibration did not use the same expected support identity")
    require(sha256(cache["token_path"]) == cache["token_sha256"], "Support cache checksum mismatch")
    tokens = np.load(cache["token_path"], mmap_mode="r", allow_pickle=False)
    require(tokens.shape == tuple(cache["shape"]) == (support_count, 1370, 768) and
            tokens.dtype == np.float32, "Wrong frozen DINO token shape/dtype")
    require(all(np.isfinite(tokens[i:i + 8]).all() for i in range(0, len(tokens), 8)),
            "Nonfinite frozen DINO tokens")
    del tokens
    checkpoint_path = directory / "i2cl.pt"
    require(sha256(checkpoint_path) == result["checkpoint_sha256"], "Checkpoint checksum mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(checkpoint["run_config"] == checkpoint["config"] == config, "Checkpoint run config differs")
    require(checkpoint["run_fingerprint"] == config["fingerprint"], "Checkpoint fingerprint differs")
    require(checkpoint["model_identity"] == config["checkpoint"], "Checkpoint base-model identity differs")
    require(checkpoint["dataset"] == name and checkpoint["class_names"] == support["class_names"]
            and checkpoint["prompts"] == support["prompts"], "Checkpoint dataset/prompts differ")
    require(checkpoint["support_sha256"] == support["metadata"]["support_sha256"],
            "Checkpoint support CSV hash differs")
    require(all(checkpoint[key] == support_identity for key in
                ("support_identity", "context_support_identity", "calibration_support_identity")),
            "Checkpoint support identities differ")
    state = checkpoint["adaptation_state"]
    require(state["context_pooling"] == "cls" and state["layer_count"] == 2 and
            state["hidden_dim"] == 768 and state["module_names"] == ["attention", "mlp"],
            "Unexpected I2CL architecture")
    require(state["context_sample_count"] == support_count, "Wrong context sample count")
    coefficients, context = state["coefficients"], state["context_vectors"]
    require(tuple(coefficients.shape) == (2, 2, 2) and coefficients.numel() == 8 and
            tuple(context.shape) == (2, 2, 768) and context.numel() == 3072,
            "Wrong coefficient/context shape")
    require(torch.isfinite(coefficients).all().item() and torch.isfinite(context).all().item(),
            "Nonfinite saved coefficients/context")
    require(config["trainable_scalars"] == result["trainable_scalars"] == 8, "Wrong trainable scalar count")
    training = checkpoint["training"]
    require(training == result["training"], "Saved training records differ")
    loss_weighting = verify_loss_weighting(config, training, support)
    context_sha = hashlib.sha256(context.contiguous().numpy().tobytes()).hexdigest()
    require(training["context_sha256"] == context_sha and training["context_unchanged"],
            "Saved fixed context hash differs")
    gradients = np.asarray(training["first_gradient"], dtype=np.float64)
    require(gradients.shape == (2, 2, 2) and np.isfinite(gradients).all(),
            "Invalid eight-coefficient gradient record")
    require(training["all_eight_coefficients_received_nonzero_finite_gradients"] and
            training["base_parameters_received_gradients"] is False,
            "Runtime gradient isolation check failed")
    history = read_json(directory / "train_history.json")
    regularization = verify_regularization(config, training, history)
    require(len(history) == args["epochs"] == training["epochs"], "Wrong completed epoch count")
    require(training["examples_per_epoch"] == support_count and
            training["optimizer_steps"] == args["epochs"] *
            math.ceil(support_count / args["grad_batch_size"]), "Wrong calibration sample/step totals")
    for index, epoch in enumerate(history, start=1):
        require(epoch["epoch"] == index and epoch["examples_seen"] == support_count,
                "An epoch did not consume the entire fixed support set")
        require(epoch["optimizer_steps"] == math.ceil(support_count / args["grad_batch_size"]),
                "Wrong per-epoch optimizer step count")
        require(math.isfinite(epoch["train_bce_with_noise"]), "Nonfinite training loss")
        require(not any("test" in key or "valid" in key or "best" in key for key in epoch),
                "History contains evaluation-driven selection fields")
        values = np.asarray(epoch["coefficients"], dtype=np.float32)
        require(values.shape == (2, 2, 2) and np.isfinite(values).all(), "Invalid epoch coefficients")
    require(np.array_equal(coefficients.numpy(), np.asarray(history[-1]["coefficients"], dtype=np.float32))
            and np.array_equal(coefficients.numpy(),
                               np.asarray(training["final_coefficients"], dtype=np.float32)),
            "Checkpoint coefficients are not the final epoch coefficients")
    require(config["checkpoint_selection"] == "final epoch fixed before testing" and
            training["checkpoint_selection"] == "final fixed epoch; no validation/test selection",
            "Checkpoint selection protocol changed")
    frozen = read_json(directory / "frozen_model_check.json")
    require(frozen == checkpoint["frozen_model_check"] == result["checks"]["base_unchanged"],
            "Frozen-model records disagree")
    require(frozen["before_sha256"] == frozen["after_sha256"] and
            len(frozen["before_sha256"]) == 64 and frozen["trainable_parameter_count"] == 8 and
            frozen["frozen_parameter_count"] > 1_000_000, "Frozen tensor hash/count mismatch")
    reference = read_json(directory / "reference_equivalence.json")
    require(reference == result["checks"]["reference_equivalence"] and reference["passed"] and
            reference["alpha"] == 0 and reference["beta"] == 1, "Identity-reference record differs")
    require({item["batch_size"] for item in reference["checks"]} == {1, 2},
            "Missing batch-1/batch-2 identity checks")
    for item in reference["checks"]:
        require(all(math.isfinite(item[key]) and item[key] >= 0
                    for key in ("max_logit_error", "max_map_error")), "Invalid reference error record")
    reload_check = read_json(directory / "checkpoint_reload_check.json")
    require(reload_check == result["checks"]["checkpoint_reload"] and reload_check["passed"],
            "Checkpoint reload record differs")
    require(all(math.isfinite(value) and 0 <= value <= 1e-6
                for value in reload_check["max_absolute_errors"].values()),
            "Checkpoint roundtrip output mismatch")
    for filename, hash_key in (("predictions.csv", "predictions_sha256"),
                               ("baseline_predictions.csv", "baseline_predictions_sha256")):
        require(sha256(directory / filename) == result[hash_key], f"{filename} checksum mismatch")
    logits = read_predictions(directory / "predictions.csv", spec)
    baseline = read_predictions(directory / "baseline_predictions.csv", spec)
    baseline_verification = read_json(directory / "baseline_verification.json")
    require(baseline_verification == result["checks"]["recorded_baseline"],
            "Baseline runtime verification records differ")
    verify_baseline(config, baseline, spec, original, baseline_verification)
    measured = independent_metrics(spec["labels"], logits, spec["class_names"])
    original_metrics = independent_metrics(spec["labels"], baseline, spec["class_names"])
    verify_metrics(read_json(directory / "metrics.json"), measured, "metrics.json")
    verify_metrics(result["metrics"], measured, "result.metrics")
    verify_metrics(read_json(directory / "baseline_metrics.json"), original_metrics, "baseline_metrics.json")
    verify_metrics(result["baseline_metrics"], original_metrics, "result.baseline_metrics")
    if not config["is_smoke_test"]:
        require(measured["macro_auroc_defined_classes"] == len(spec["class_names"]) and
                measured["macro_auprc_defined_classes"] == len(spec["class_names"]),
                "Full-run metrics silently omit a class")
    deltas = {}
    for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc"):
        deltas[key] = (None if measured[key] is None or original_metrics[key] is None else
                       100 * (measured[key] - original_metrics[key]))
        close(result["delta_pp"][key], deltas[key], f"result.delta_pp.{key}")
    return {
        "status": "passed",
        "dataset": name,
        "is_smoke_test": config["is_smoke_test"],
        "support_images": support_count,
        "test_images": test_count,
        "full_holdout_images": len(full_test["image_paths"]),
        "protocol": split_manifest["protocol"],
        "physical_gpu": physical_gpu,
        "epochs": args["epochs"],
        "optimizer_steps": training["optimizer_steps"],
        "shots_per_class": split_manifest["shots_per_class"],
        "learning_rate": args["learning_rate"],
        "weight_decay": args["weight_decay"],
        "noise_scale": args["noise_scale"],
        "grad_batch_size": args["grad_batch_size"],
        "sampling": config["sampling"],
        "loss": config["loss"],
        "loss_weighting": loss_weighting,
        "objective": config.get("objective", config["loss"]),
        "regularization": regularization,
        "class_names": support["class_names"],
        "baseline_mode": config["baseline"]["mode"],
        "protocol_exception": split_manifest.get("protocol_exception"),
        "support_subset_verification": subset_verification,
        "trainable_scalars": 8,
        "support_identity": support_identity,
        "support_sha256": support["metadata"]["support_sha256"],
        "same_support_context_and_calibration": True,
        "first_gradient_nonzero_count": int(np.count_nonzero(gradients)),
        "base_tensor_digest": frozen["before_sha256"],
        "context_tensor_sha256": context_sha,
        "baseline_metrics": original_metrics,
        "i2cl_metrics": measured,
        "delta_pp": deltas,
        "limitations": split_manifest["limitations"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def audit(output, allow_smoke=False):
    output = Path(output).expanduser().resolve()
    launcher = read_json(output / "launcher.json")
    require(launcher["status"] == "complete", "Launcher has not completed")
    require(not launcher["is_smoke"] or allow_smoke, "Smoke run requires --allow-smoke")
    require(set(launcher["jobs"]) == set(launcher["datasets"]), "Launcher job coverage mismatch")
    for path, expected in launcher["source_sha256"].items():
        require(sha256(path) == expected, f"Launcher source changed: {path}")
    results, failures = {}, {}
    for name in launcher["datasets"]:
        job = launcher["jobs"][name]
        try:
            require(job["status"] == "complete" and job["exit_code"] == 0, "Job did not exit successfully")
            require(job["gpu"] in launcher["physical_gpus"],
                    "Worker used a GPU outside the launch request")
            results[name] = audit_dataset(output / name, job, allow_smoke)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    report = {
        "status": "failed" if failures else "passed",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "audit_code_sha256": sha256(__file__),
        "scope": (
            "CPU recomputation of classification metrics, labels, split isolation, source hashes, "
            "checkpoint coefficients/context and final-epoch selection. Runtime CUDA freeze, gradient "
            "and reload records are cross-checked; GPU execution is not repeated by this audit."
        ),
        "datasets": results,
        "failures": failures,
        "launcher_elapsed_seconds": launcher["elapsed_seconds"],
    }
    write_json(output / "audit.json", report)
    if failures:
        raise ValueError(f"I2CL audit failed: {failures}")
    summary = {
        "status": "complete_and_audited",
        "is_smoke_test": launcher["is_smoke"],
        "metric_units": "AUROC/AUPRC as fractions; delta_pp as percentage points",
        "auprc_method": "trapezoidal auc(recall, precision), not average precision",
        "datasets": results,
        "elapsed_seconds": launcher["elapsed_seconds"],
        "audit": str(output / "audit.json"),
    }
    write_json(output / "summary.json", summary)
    fields = [
        "dataset", "support_images", "test_images", "protocol", "physical_gpu",
        "baseline_macro_auroc_pct", "i2cl_macro_auroc_pct", "delta_auroc_pp",
        "baseline_macro_auprc_pct", "i2cl_macro_auprc_pct", "delta_auprc_pp",
        "epochs", "trainable_scalars", "is_smoke_test",
    ]
    temporary = output / "summary.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in launcher["datasets"]:
            row = results[name]
            item = {key: row[key] for key in
                    ("dataset", "support_images", "test_images", "protocol", "physical_gpu",
                     "epochs", "trainable_scalars", "is_smoke_test")}
            for metric in ("auroc", "auprc"):
                for method in ("baseline", "i2cl"):
                    value = row[f"{method}_metrics"][f"macro_{metric}"]
                    item[f"{method}_macro_{metric}_pct"] = None if value is None else 100 * value
                item[f"delta_{metric}_pp"] = row["delta_pp"][f"macro_{metric}"]
            writer.writerow(item)
    temporary.replace(output / "summary.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    report = audit(args.output, args.allow_smoke)
    print(json.dumps({"status": report["status"], "output": report["output"],
                      "datasets": list(report["datasets"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
