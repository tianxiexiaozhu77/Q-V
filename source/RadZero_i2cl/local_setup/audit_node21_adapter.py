#!/usr/bin/env python3
"""Independently audit a completed Node21 adapter run using CPU artifacts only.

AUROC and trapezoidal PR AUC are recomputed directly from saved logits, without
calling the training metrics helper. Cache token payloads are not re-inferred.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np
from scipy.special import expit
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from infer import ASSETS_DIR, MODEL_DIR
from node21_adapter_data import load_split, validate_splits


SPLITS = ("train", "valid", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-12):
    require(
        isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
        f"{message}: {actual!r} != {expected!r}",
    )


def metric_values(labels, logits):
    labels = np.asarray(labels).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    require(labels.shape == logits.shape and labels.size > 0, "Metric array sizes differ or are empty")
    require(set(np.unique(labels)) == {0, 1}, "AUROC audit requires both target classes")
    require(np.isfinite(logits).all(), "Nonfinite logits in saved predictions")
    precision, recall, _ = precision_recall_curve(labels, logits)
    return {
        "images": int(labels.size),
        "positive_images": int(labels.sum()),
        "negative_images": int(labels.size - labels.sum()),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(auc(recall, precision)),
    }


def check_metric_record(record, expected, title):
    require(record["dataset"] == "Node21", title + ": unexpected dataset")
    require(record["sample_count"] == expected["images"], title + ": sample count differs")
    require(record["class_count"] == 1, title + ": class count differs")
    for scope in ("macro", "micro"):
        for metric in ("auroc", "auprc"):
            close(record[f"{scope}_{metric}"], expected[metric], f"{title}: {scope}_{metric}")
    cls = record["per_class"]["Nodule"]
    for metric in ("auroc", "auprc"):
        close(cls[metric], expected[metric], f"{title}: Nodule {metric}")
    require(cls["positives"] == expected["positive_images"], title + ": positive count differs")
    require(cls["negatives"] == expected["negative_images"], title + ": negative count differs")


def read_predictions(path, spec):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len(spec["image_paths"]), f"{path}: incorrect prediction count")
    require([int(row["index"]) for row in rows] == list(range(len(rows))), f"{path}: invalid indices")
    require([row["image_path"] for row in rows] == spec["image_paths"], f"{path}: image paths/order differ")
    labels = np.array([int(row["label:Nodule"]) for row in rows])
    require(np.array_equal(labels, spec["labels"].ravel()), f"{path}: labels differ")
    logits = np.array([float(row["logit:Nodule"]) for row in rows])
    scores = np.array([float(row["score:Nodule"]) for row in rows])
    require(np.isfinite(logits).all() and np.isfinite(scores).all(), f"{path}: nonfinite predictions")
    np.testing.assert_allclose(scores, expit(logits), atol=1e-12, rtol=1e-12)
    return logits, metric_values(labels, logits)


def read_original_baseline(path, spec):
    """Join the earlier complete public-data run by absolute image path."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require([int(row["index"]) for row in rows] == list(range(len(rows))),
            "Original baseline indices are not complete and consecutive")
    by_path = {str(Path(row["image_path"]).resolve()): row for row in rows}
    require(len(by_path) == len(rows) == 4882,
            "Original baseline must contain all 4,882 unique public images")
    selected = []
    for path, target in zip(spec["image_paths"], spec["labels"].ravel()):
        require(str(Path(path).resolve()) in by_path, f"Baseline lacks {path}")
        row = by_path[str(Path(path).resolve())]
        require(int(row["label:Nodule"]) == int(target), f"Baseline label differs for {path}")
        selected.append(row)
    logits = np.asarray([float(row["logit:Nodule"]) for row in selected])
    scores = np.asarray([float(row["score:Nodule"]) for row in selected])
    np.testing.assert_allclose(scores, expit(logits), atol=1e-12, rtol=1e-12)
    return logits, metric_values(spec["labels"], logits)


def validate_real_targets(specs):
    """Read released real annotations directly, without the split loader."""
    parents = {Path(path).parent for spec in specs.values() for path in spec["image_paths"]}
    require(len(parents) == 1, "Expected one released processed MHA image directory")
    image_root = parents.pop()
    annotation_path = image_root.parent / "metadata.csv"
    with annotation_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {}
    box_counts = {}
    for row in rows:
        name = row["img_name"]
        target = int(row["label"])
        require(target in (0, 1) and Path(name).name == name, "Invalid real annotation")
        require(name not in labels or labels[name] == target, "Inconsistent real image labels")
        labels[name] = target
        if target:
            require(float(row["width"]) > 0 and float(row["height"]) > 0,
                    "Positive real annotation lacks a nodule box")
            box_counts[name] = box_counts.get(name, 0) + 1
    require(len(labels) == 4882 and sum(labels.values()) == 1134 and len(rows) == 5224,
            "Unexpected public Node21 real annotation counts")
    for name, spec in specs.items():
        for path, image_id, target in zip(spec["image_paths"], spec["ids"], spec["labels"].ravel()):
            path = Path(path)
            require(path.stem == image_id and path.name in labels,
                    f"{name}: unknown or inconsistent real image identifier")
            require(int(target) == labels[path.name], f"{name}: target differs from real annotation")
    return {
        "path": str(annotation_path), "sha256": sha256(annotation_path),
        "real_images": len(labels), "positive_images": sum(labels.values()),
        "real_box_rows": sum(box_counts.values()), "simulated_metadata_used": False,
    }


def validate_group_records(split_dir, specs):
    """Verify all persisted identity/hash groups before any smoke truncation."""
    fields = (
        "group_id", "patient_id", "study_id",
        "file_sha256", "pixel_sha256", "normalized_uint8_sha256",
    )
    records = {}
    for name in SPLITS:
        with (split_dir / (name + ".csv")).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == len(specs[name]["image_paths"]), f"{name}: split CSV row count differs")
        require([row["image_path"] for row in rows] == specs[name]["image_paths"],
                f"{name}: split CSV image ordering differs")
        require([row["image_id"] for row in rows] == specs[name]["ids"],
                f"{name}: split CSV identifiers differ")
        require(np.array_equal([int(row["label"]) for row in rows], specs[name]["labels"].ravel()),
                f"{name}: split CSV labels differ")
        for row in rows:
            require(all(field in row for field in fields), f"{name}: missing group/hash fields")
            for field in ("file_sha256", "pixel_sha256", "normalized_uint8_sha256"):
                value = row[field]
                require(len(value) == 64 and set(value) <= set("0123456789abcdef"),
                        f"{name}: invalid recorded {field}")
            require(bool(row["group_id"]), f"{name}: empty duplicate/patient group")
        records[name] = rows
    intersections = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        pair = {}
        for field in fields:
            left_values = {row[field] for row in records[left] if row[field]}
            right_values = {row[field] for row in records[right] if row[field]}
            pair[field] = len(left_values & right_values)
            require(pair[field] == 0, f"Full splits overlap in {field}: {left}, {right}")
        intersections[left + "_" + right] = pair
    return intersections


def official_scoring_digest():
    from safetensors import safe_open
    prefix = "loss_fns.KeyPhraseAlignmentLoss."
    digest = hashlib.sha256()
    with safe_open(MODEL_DIR / "model.safetensors", framework="pt", device="cpu") as weights:
        keys = sorted(key for key in weights.keys() if key.startswith(prefix))
        require(bool(keys), "Official checkpoint has no scoring module parameters")
        for key in keys:
            digest.update(key[len(prefix):].encode())
            digest.update(weights.get_tensor(key).contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_reference_run(reference_run, manifest, summary, cache_manifest):
    """Bind extraction checks to the completed run that created this cache."""
    reference_run = Path(reference_run).expanduser().resolve()
    source = read_json(reference_run / "run_manifest.json")
    source_summary = read_json(reference_run / "summary.json")
    source_progress = read_json(reference_run / "progress.json")
    source_launcher = read_json(reference_run / "launcher.json")
    require(Path(source["paths"]["output"]).resolve() == reference_run,
            "Cache-reference run path differs from its manifest")
    source_fingerprint = fingerprint({
        key: value for key, value in source.items() if key not in ("started_at", "fingerprint")
    })
    require(source_fingerprint == source["fingerprint"] == source_summary["run_fingerprint"],
            "Cache-reference run identity fingerprint mismatch")
    require(source_summary["status"] == source_progress["status"] == summary["status"],
            "Cache-reference run is not complete for the same experiment scope")
    require(source_launcher["status"] == "complete" and source_launcher["exit_code"] == 0,
            "Cache-reference run launcher did not complete successfully")
    require(source_summary["cache_fingerprint"] == cache_manifest["fingerprint"],
            "Cache-reference run used another feature cache fingerprint")
    require(source_summary["counts"] == summary["counts"] == cache_manifest["counts"],
            "Cache-reference run used different split counts")
    for key in ("cache_dir", "splits_dir"):
        require(Path(source["paths"][key]).resolve() == Path(manifest["paths"][key]).resolve(),
                f"Cache-reference run used another {key}")
    for key in ("code_sha256", "checkpoint", "split_manifest_sha256", "world_size",
                "physical_gpus", "architecture", "prompt", "precision"):
        require(source[key] == manifest[key], f"Cache-reference run differs in {key}")
    require(source["hyperparameters"]["extract_batch_size"]
            == manifest["hyperparameters"]["extract_batch_size"],
            "Cache-reference run used another extraction batch size")
    return {
        "output": str(reference_run),
        "run_fingerprint": source_fingerprint,
        "manifest_sha256": sha256(reference_run / "run_manifest.json"),
        "summary_sha256": sha256(reference_run / "summary.json"),
    }


def audit(output, cache_reference_run=None):
    output = Path(output).expanduser().resolve()
    manifest = read_json(output / "run_manifest.json")
    summary = read_json(output / "summary.json")
    history = read_json(output / "history.json")
    launcher = read_json(output / "launcher.json")
    progress = read_json(output / "progress.json")
    hyper = manifest["hyperparameters"]
    limits = {name: int(hyper["limit_" + name]) for name in SPLITS}
    limited = any(limits.values())
    expected_status = "smoke_complete" if limited else "complete"
    require(summary["status"] == expected_status, "Run summary is not successfully complete")
    require(progress["status"] == expected_status, "Run progress is not successfully complete")
    require(launcher["status"] == "complete" and launcher["exit_code"] == 0, "Launcher has not completed successfully")
    require(manifest["world_size"] == 2, "This experiment requires two distributed ranks")
    require(manifest["physical_gpus"] == "0,1", "Run manifest did not use physical GPUs 0 and 1")
    require(launcher["gpus"] == [0, 1] and launcher["cuda_visible_devices"] == "0,1",
            "Launcher GPU assignment differs")
    require(Path(manifest["paths"]["output"]).resolve() == output, "Run output path differs from manifest")
    original_fingerprint = fingerprint({
        key: value for key, value in manifest.items() if key not in ("started_at", "fingerprint")
    })
    require(manifest["fingerprint"] == original_fingerprint == summary["run_fingerprint"],
            "Run identity fingerprint mismatch")
    for filename, expected_sha in manifest["code_sha256"].items():
        require(sha256(Path(__file__).with_name(filename)) == expected_sha,
                f"Recorded implementation changed after run: {filename}")
    for filename, expected_sha in manifest["model_source_sha256"].items():
        require(sha256(MODEL_DIR / filename) == expected_sha,
                f"Official model implementation changed after run: {filename}")
    for filename, expected_sha in manifest["tokenizer_sha256"].items():
        require(sha256(MODEL_DIR / filename) == expected_sha,
                f"Official tokenizer file changed after run: {filename}")
    for key, filename in (
        ("model_config_sha256", "config.json"),
        ("preprocessor_sha256", "preprocessor_config.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
    ):
        require(sha256(MODEL_DIR / filename) == manifest[key], f"Official {filename} changed")
    weight_sha = sha256(MODEL_DIR / "model.safetensors")
    require(weight_sha == manifest["checkpoint"]["weight_sha256"], "Official model file was modified")

    split_dir = Path(manifest["paths"]["splits_dir"])
    split_manifest = validate_splits(split_dir)
    require(sha256(split_dir / "split_manifest.json") == manifest["split_manifest_sha256"],
            "Run split manifest differs")
    for name in SPLITS:
        require(sha256(split_dir / (name + ".csv")) == manifest["split_csv_sha256"][name],
                f"{name} split CSV differs from training identity")
    specs = {name: load_split(split_dir, name) for name in SPLITS}
    real_targets = validate_real_targets(specs)
    full_group_intersections = validate_group_records(split_dir, specs)
    selected_full_paths = [path for spec in specs.values() for path in spec["image_paths"]]
    require(len(selected_full_paths) == len(set(selected_full_paths)) == 4882,
            "Full split collection must partition all 4,882 public images exactly once")
    for name, spec in specs.items():
        full_count = len(spec["image_paths"])
        count = min(limits[name], full_count) if limits[name] else full_count
        for key, values in list(spec.items()):
            if key not in ("class_names", "prompts", "source_files") and isinstance(values, (list, np.ndarray)):
                if len(values) == full_count:
                    spec[key] = values[:count]
        require(summary["counts"][name] == count, f"Run count differs from selected {name} split")
        require(summary["class_counts"][name] == {
            "positive": int(spec["labels"].sum()), "negative": int(count - spec["labels"].sum())
        }, f"Run class counts differ from selected {name} split")
    intersections = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        pair = {}
        for key in ("ids", "image_paths", "group_ids", "patient_ids", "study_ids"):
            require(key in specs[left] and key in specs[right], f"Missing split grouping field {key}")
            pair[key] = len(set(filter(None, specs[left][key])) & set(filter(None, specs[right][key])))
            require(pair[key] == 0, f"Cross-split {key} leakage: {left}, {right}")
        intersections[f"{left}_{right}"] = pair

    cache = Path(manifest["paths"]["cache_dir"])
    cache_manifest = read_json(cache / "manifest.json")
    cache_payload = {
        key: value for key, value in cache_manifest.items()
        if key not in ("fingerprint", "status", "created_at", "completed_at", "total_images", "tokens_bytes")
    }
    require(fingerprint(cache_payload) == cache_manifest["fingerprint"] == summary["cache_fingerprint"],
            "Feature cache fingerprint mismatch")
    require(cache_manifest["status"] == "complete", "Feature cache is incomplete")
    require(cache_manifest["counts"] == summary["counts"], "Feature-cache counts differ")
    require(cache_manifest["split_manifest_sha256"] == manifest["split_manifest_sha256"],
            "Feature cache used another split manifest")
    require(cache_manifest["checkpoint"] == manifest["checkpoint"], "Feature cache used another checkpoint")
    for key in ("code_sha256", "model_source_sha256", "tokenizer_sha256", "split_csv_sha256", "prompt"):
        require(cache_manifest[key] == manifest[key], f"Cache identity differs in {key}")
    require(cache_manifest["extractor_sha256"] == manifest["code_sha256"]["train_node21_adapter.py"]
            and cache_manifest["adapter_code_sha256"] == manifest["code_sha256"]["siim_adapter_model.py"],
            "Feature cache used another extraction or adapter implementation")
    require(cache_manifest["preprocessor_sha256"] == sha256(MODEL_DIR / "preprocessor_config.json")
            and cache_manifest["model_config_sha256"] == sha256(MODEL_DIR / "config.json"),
            "Feature cache model or preprocessing configuration differs")
    reference_provenance = {
        "cache_fingerprint": cache_manifest["fingerprint"],
        "fallback_run": None,
        "files": [],
    }
    if cache_reference_run is not None:
        reference_provenance["fallback_run"] = validate_reference_run(
            cache_reference_run, manifest, summary, cache_manifest)
    for name, spec in specs.items():
        require(cache_manifest["paths_and_labels"][name] == fingerprint({
            "paths": spec["image_paths"], "labels": spec["labels"].tolist()
        }), f"Feature cache {name} paths or labels differ")
        tokens = np.load(cache / f"{name}_tokens.npy", mmap_mode="r", allow_pickle=False)
        require(tokens.shape == (len(spec["image_paths"]), 1370, 768) and tokens.dtype == np.float32,
                f"Invalid cached {name} token shape or dtype")
        del tokens

    ranks = []
    for rank in (0, 1):
        completion = read_json(output / f"rank{rank}/completion.json")
        require(completion["status"] == "complete" and completion["rank"] == rank,
                f"Rank {rank} did not finish")
        require(completion["cuda_visible_devices"] == "0,1" and completion["device"] == f"cuda:{rank}",
                f"Rank {rank} used unexpected GPU mapping")
        require(not (output / f"rank{rank}_error.json").exists(), f"Rank {rank} has an error artifact")
        cache_progress = read_json(cache / f"rank{rank}_progress.json")
        require(cache_progress["status"] == "complete" and cache_progress["rank"] == rank,
                f"Rank {rank} feature extraction is incomplete")
        require(cache_progress["fingerprint"] == cache_manifest["fingerprint"],
                f"Rank {rank} feature cache identity differs")
        for name, spec in specs.items():
            count = len(spec["image_paths"])
            start, end = count * rank // 2, count * (rank + 1) // 2
            entry = cache_progress["splits"][name]
            require(
                entry["start"] == start and entry["end"] == end and entry["end_index"] == end
                and entry["completed_images"] == end - start and entry["total_images"] == end - start,
                f"Rank {rank} {name} extraction range is incomplete",
            )
            reference_path = output / f"rank{rank}/{name}_identity_reference.json"
            reference_output = output
            if not reference_path.exists():
                require(reference_provenance["fallback_run"] is not None,
                        f"{reference_path} is missing; for a reused cache, explicitly supply "
                        "--cache-reference-run pointing to the completed extraction run")
                reference_output = Path(reference_provenance["fallback_run"]["output"])
                reference_path = reference_output / f"rank{rank}/{name}_identity_reference.json"
            reference = read_json(reference_path)
            require(reference["passed"] is True and bool(reference["checks"]),
                    f"Rank {rank} {name} identity checks did not pass")
            for check in reference["checks"]:
                require(
                    math.isfinite(check["max_logit_error"]) and math.isfinite(check["max_map_error"])
                    and check["max_logit_error"] >= 0 and check["max_map_error"] >= 0,
                    "Invalid reported reference error",
                )
            reference_provenance["files"].append({
                "rank": rank,
                "split": name,
                "path": str(reference_path),
                "sha256": sha256(reference_path),
                "source_run": str(reference_output),
                "reused_from_another_run": reference_output != output,
            })
        ranks.append(completion)

    baseline_valid_logits, baseline_valid = read_predictions(
        output / "baseline_validation_predictions.csv", specs["valid"])
    best_valid_logits, best_valid = read_predictions(
        output / "best_validation_predictions.csv", specs["valid"])
    test_logits, test = read_predictions(output / "test_predictions.csv", specs["test"])
    original_baseline_path = ASSETS_DIR / "outputs/node21_zeroshot_gpu01/predictions.csv"
    require(Path(manifest["baseline_source"]).resolve() == original_baseline_path.resolve()
            and sha256(original_baseline_path) == manifest["baseline_source_sha256"],
            "Earlier baseline source path or contents changed")
    require(sha256(original_baseline_path.parent / "run_manifest.json")
            == manifest["baseline_run_manifest_sha256"], "Earlier baseline manifest changed")
    original_baseline_manifest = read_json(original_baseline_path.parent / "run_manifest.json")
    require(original_baseline_manifest["checkpoint"] == manifest["checkpoint"]
            and original_baseline_manifest["prompts"] == [manifest["prompt"]]
            and original_baseline_manifest["class_names"] == ["Nodule"],
            "Earlier baseline used another original checkpoint, class, or prompt")
    for key in ("model_config_sha256", "preprocessor_sha256"):
        require(original_baseline_manifest[key] == manifest[key],
                f"Earlier baseline differs in {key}")
    original_test_logits, original_test = read_original_baseline(original_baseline_path, specs["test"])
    baseline_test_logits = np.load(cache / "test_baseline_logits.npy", allow_pickle=False).reshape(-1)
    baseline_test = metric_values(specs["test"]["labels"], baseline_test_logits)
    saved_baseline_test_logits, _ = read_predictions(output / "baseline_test_predictions.csv", specs["test"])
    np.testing.assert_array_equal(saved_baseline_test_logits, baseline_test_logits)
    np.testing.assert_allclose(baseline_test_logits, original_test_logits, atol=2e-5, rtol=1e-4)
    np.testing.assert_array_equal(
        baseline_valid_logits, np.load(cache / "valid_baseline_logits.npy", allow_pickle=False).reshape(-1))
    original_valid_logits, _ = read_original_baseline(original_baseline_path, specs["valid"])
    np.testing.assert_allclose(baseline_valid_logits, original_valid_logits, atol=2e-5, rtol=1e-4)
    for filename, values in (
        ("baseline_validation_metrics.json", baseline_valid),
        ("best_validation_metrics.json", best_valid),
        ("baseline_test_metrics.json", baseline_test),
        ("test_metrics.json", test),
    ):
        check_metric_record(read_json(output / filename), values, filename)
    check_metric_record(summary["validation"], best_valid, "summary validation")
    check_metric_record(summary["test"], test, "summary test")
    baseline_reference = read_json(output / "test_baseline_equivalence.json")
    require(baseline_reference == summary["baseline_test_equivalence"], "Baseline-equivalence records differ")
    require(baseline_reference["passed"] and baseline_reference["images"] == len(test_logits),
            "Original baseline equivalence did not pass")
    close(baseline_reference["max_logit_error"], float(np.max(np.abs(baseline_test_logits-original_test_logits))),
          "Original baseline maximum logit error")

    require(history == summary["history"], "Summary and history.json differ")
    require(len(history) == summary["epochs_completed"] and len(history) > 0, "Invalid epoch count")
    best_score, best_epoch, stale = -math.inf, 0, 0
    patience, maximum_epochs = int(hyper["patience"]), int(hyper["epochs"])
    padded_examples = math.ceil(len(specs["train"]["image_paths"]) / 2) * 2
    for epoch, entry in enumerate(history, start=1):
        require(entry["epoch"] == epoch, "Training epochs are not consecutive")
        require(stale < patience, "Training continued beyond its patience stopping criterion")
        score = float(entry["valid_auroc"])
        require(math.isfinite(score) and 0 <= score <= 1, "Invalid historical validation AUROC")
        require(math.isfinite(entry["valid_auprc"]) and 0 <= entry["valid_auprc"] <= 1,
                "Invalid historical validation AUPRC")
        require(math.isfinite(entry["train_weighted_bce"]) and entry["train_weighted_bce"] >= 0,
                "Invalid training loss")
        require(entry["train_examples_including_distributed_padding"] == padded_examples,
                "Training epoch sample count differs from distributed sampler")
        selected = score > best_score + 1e-4
        if selected:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        require(entry["selected"] == selected and entry["best_epoch"] == best_epoch,
                f"Epoch {epoch}: validation selection differs from min_delta=0.0001 rule")
        close(entry["best_valid_auroc"], best_score, f"Epoch {epoch}: best validation AUROC")
        gradient = entry["gradient_check"]
        require(gradient["frozen_gradients_none"] is True and gradient["adapter_gradient_norm"] > 0
                and math.isfinite(gradient["adapter_gradient_norm"]), "Invalid training gradient check")
    require(len(history) <= maximum_epochs, "More epochs than configured")
    require(len(history) == maximum_epochs or stale >= patience, "Run stopped before max epochs without patience")
    require(summary["early_stopped"] == (len(history) < maximum_epochs), "Incorrect early-stop flag")
    require(summary["best_epoch"] == best_epoch, "Selected checkpoint epoch differs from validation history")
    close(best_valid["auroc"], best_score, "Selected validation prediction AUROC")
    close(best_valid["auprc"], history[best_epoch-1]["valid_auprc"], "Selected validation prediction AUPRC")

    import torch
    best = torch.load(output / "best_adapter.pt", map_location="cpu", weights_only=True)
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    require(best["epoch"] == best_epoch and best["run_fingerprint"] == manifest["fingerprint"],
            "Saved best checkpoint identity/epoch differs")
    check_metric_record(best["validation_metrics"], best_valid, "best checkpoint validation")
    require(best["checkpoint"] == manifest["checkpoint"], "Adapter references another original checkpoint")
    require(best["prompt"] == manifest["prompt"] == "There is Lung Nodule",
            "Adapter uses another prompt")
    require(best["architecture"] == {"hidden_dim": 768, "bottleneck": 32, "scale": 0.1},
            "Saved adapter architecture differs")
    require(last["history"] == history and last["epoch"] == len(history) and last["best_epoch"] == best_epoch
            and last["stale"] == stale and last["run_fingerprint"] == manifest["fingerprint"],
            "Final resume checkpoint differs from training history")
    close(last["best_score"], best_score, "Final checkpoint best score")
    state = best["adapter_state_dict"]
    expected_shapes = {"down.weight": (32, 768), "down.bias": (32,), "up.weight": (768, 32), "up.bias": (768,)}
    require(set(state) == set(expected_shapes), "Best checkpoint contains unexpected adapter parameters")
    for key, shape in expected_shapes.items():
        require(tuple(state[key].shape) == shape and bool(torch.isfinite(state[key]).all()),
                f"Invalid adapter parameter: {key}")
    trainable_count = sum(tensor.numel() for tensor in state.values())
    require(trainable_count == 49952 == summary["parameters"]["trainable"], "Incorrect trainable parameter count")
    require(bool(torch.count_nonzero(state["up.weight"])) or bool(torch.count_nonzero(state["up.bias"])),
            "Best adapter remained at its initial identity")
    require(summary["original_model_frozen"] is True, "Run does not report a frozen original model")
    official_digest = official_scoring_digest()
    require(summary["frozen_scoring_sha256_before"] == summary["frozen_scoring_sha256_after"] == official_digest,
            "Frozen scoring module hashes differ from each other or official checkpoint")

    expected_comparison = []
    for name, baseline, adapted in (("valid", baseline_valid, best_valid), ("test", baseline_test, test)):
        row = {"split": name, "images": len(specs[name]["image_paths"])}
        for key in ("auroc", "auprc"):
            row["baseline_" + key] = baseline[key]
            row["adapter_" + key] = adapted[key]
            row["delta_" + key + "_percentage_points"] = (adapted[key] - baseline[key]) * 100
        expected_comparison.append(row)
    with (output / "comparison.csv").open(newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    require(len(comparison_rows) == len(summary["comparison"]) == 2, "Comparison should contain valid and test")
    for expected, csv_row, saved in zip(expected_comparison, comparison_rows, summary["comparison"]):
        require(saved == expected, "Summary comparison differs from recomputed metrics")
        require(csv_row["split"] == expected["split"] and int(csv_row["images"]) == expected["images"],
                "Comparison CSV split/count differs")
        for key in expected.keys() - {"split", "images"}:
            close(float(csv_row[key]), expected[key], "Comparison CSV " + key)

    return {
        "status": "passed",
        "errors": [],
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "scope": "limited smoke subset" if limited else
                 "Node21 supervised image-level nodule classification on a custom public-data split; "
                 "not the hidden challenge test or official detection/FROC evaluation",
        "run_fingerprint": manifest["fingerprint"],
        "split_fingerprint": split_manifest["fingerprint"],
        "counts": summary["counts"],
        "gpu_ids": [0, 1],
        "rank_completions": ranks,
        "identity_reference_provenance": reference_provenance,
        "split_intersections": intersections,
        "full_split_group_intersections": full_group_intersections,
        "real_target_source": real_targets,
        "patient_metadata_coverage": split_manifest["patient_metadata_coverage"],
        "baseline_reference": {
            "path": str(original_baseline_path), "sha256": sha256(original_baseline_path),
            "selection": "absolute-image-path join to this run's held-out test, not all-public aggregate",
        },
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": summary["early_stopped"],
        "selection_min_delta": 1e-4,
        "stale_epochs": stale,
        "comparison": expected_comparison,
        "original_baseline_test_metrics": original_test,
        "baseline_test_max_logit_error": baseline_reference["max_logit_error"],
        "official_checkpoint_sha256": weight_sha,
        "official_and_reported_frozen_scoring_sha256": official_digest,
        "trainable_parameters": trainable_count,
        "checks": [
            "Real Node21 metadata re-read directly; split labels match real nodule annotations",
            "All public image IDs and paths partitioned exactly once; known patient/study and duplicate groups disjoint",
            "Earlier zero-shot predictions joined by absolute path to the identical held-out images",
            "Saved prediction scores equal sigmoid(logits), all logits finite",
            "AUROC and trapezoidal PR AUC independently recomputed from saved predictions",
            "Validation history replayed with min_delta=0.0001 and patience; best checkpoint matches",
            "Two completed ranks mapped to physical GPU 0 and GPU 1; cache shard coverage complete",
            "Identity-check provenance recorded; external checks require an explicit compatible completed run",
            "Original checkpoint SHA256 unchanged; before/after scoring hashes match official tensor values",
            "Saved adapter parameter shapes, values and nonzero learned residual verified",
        ],
        "limitations": [
            "Patient identity is unavailable for most images, including all 3,748 negatives; "
            "known identity and duplicate grouping cannot guarantee full patient separation",
            "This is a newly created public-data split, not the hidden Node21 challenge test",
            "Full cached image-token payloads are not reread or re-inferred during this CPU audit",
            "Per-epoch validation metrics beyond the selected checkpoint are checked from recorded history",
            "Frozen during-training gradients and GPU mapping are verified from recorded runtime evidence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-reference-run", type=Path,
        help="Completed extraction run supplying missing identity checks for the identical reused cache",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    require(output.is_dir(), f"Output directory does not exist: {output}")
    try:
        result = audit(output, cache_reference_run=args.cache_reference_run)
        exit_code = 0
    except Exception as error:
        result = {
            "status": "failed",
            "errors": [repr(error)],
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "output": str(output),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    temporary = output / "audit.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output / "audit.json")
    print(json.dumps({key: result[key] for key in ("status", "errors", "output")}, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
