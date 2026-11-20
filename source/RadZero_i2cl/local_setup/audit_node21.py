#!/usr/bin/env python3
"""Independently audit saved Node21 image-level classification predictions.

This program reads the original metadata and saved predictions on CPU. It does
not import the inference data loader or metrics implementation, and does not
claim that classification results reproduce the private challenge benchmark.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT.parent / "radzero-assets" / "model"
DATA_ROOT = PROJECT_ROOT.parent / "alpharad-assets" / "data_images" / "Node21"


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
    if expected is None:
        require(actual is None, message + ": expected an undefined metric")
    else:
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
                f"{message}: {actual!r} != {expected!r}")


def metadata_images(path):
    """Aggregate box rows to image labels using first occurrence order."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    images = {}
    boxes = 0
    for row in rows:
        name = row["img_name"]
        require(Path(name).name == name and name.endswith(".mha"),
                f"Unexpected image filename: {name}")
        label = int(row["label"])
        require(label in (0, 1), "Source labels must be 0 or 1")
        if name in images:
            require(images[name] == label, f"Conflicting image labels: {name}")
        else:
            images[name] = label
        if label:
            require(float(row["width"]) > 0 and float(row["height"]) > 0,
                    f"Nonpositive nodule box size: {name}")
            boxes += 1
    require(len(rows) == 5224 and len(images) == 4882,
            "Unexpected public Node21 metadata size")
    require(sum(images.values()) == 1134 and boxes == 1476,
            "Unexpected public Node21 positive image or box count")
    return images, {"annotation_rows": len(rows), "images": len(images),
                    "positive_images": sum(images.values()),
                    "negative_images": len(images) - sum(images.values()),
                    "positive_boxes": boxes}


def metric_values(labels, logits):
    labels = np.asarray(labels).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    require(labels.shape == logits.shape and labels.size > 0, "Empty or mismatched predictions")
    require(np.isin(labels, (0, 1)).all(), "Nonbinary prediction labels")
    require(np.isfinite(logits).all(), "Nonfinite saved logits")
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    area = None
    if positive:
        precision, recall, _ = precision_recall_curve(labels, logits)
        area = float(auc(recall, precision))
    return {"images": len(labels), "positive_images": positive, "negative_images": negative,
            "auroc": float(roc_auc_score(labels, logits)) if positive and negative else None,
            "auprc": area}


def check_metrics(record, expected, class_name):
    require(record["dataset"] == "Node21", "Unexpected dataset in metrics")
    require(record["evaluation_unit"] == "image", "Unexpected evaluation unit")
    require(record["sample_count"] == expected["images"] and record["class_count"] == 1,
            "Metric sample or class count mismatch")
    require("trapezoidal" in record["auprc_method"], "Unexpected PR area definition")
    for scope in ("macro", "micro"):
        for metric in ("auroc", "auprc"):
            close(record[f"{scope}_{metric}"], expected[metric], f"{scope}_{metric}")
    cls = record["per_class"][class_name]
    for metric in ("auroc", "auprc"):
        close(cls[metric], expected[metric], f"{class_name} {metric}")
    require(cls["positives"] == expected["positive_images"]
            and cls["negatives"] == expected["negative_images"],
            "Metric class prevalence mismatch")


def source_hash_checks(manifest):
    checked = {}
    for key in ("source_sha256", "code_sha256"):
        require(bool(manifest[key]), f"Missing {key}")
        checked[key] = {}
        for filename, expected in manifest[key].items():
            path = Path(filename)
            if not path.is_absolute():
                path = PROJECT_ROOT / path if len(path.parts) > 1 else Path(__file__).with_name(filename)
            require(sha256(path) == expected, f"File changed after run: {path}")
            checked[key][str(path)] = expected
    checkpoint = manifest["checkpoint"]
    official_download = read_json(MODEL_DIR / "download_manifest.json")
    official_sha = next(item["hash"] for item in official_download["files"]
                        if item["path"] == "model.safetensors")
    actual_sha = sha256(MODEL_DIR / "model.safetensors")
    require(actual_sha == official_sha == checkpoint["weight_sha256"],
            "Checkpoint differs from official download or run identity")
    require(checkpoint["repo"] == official_download["repo"]
            and checkpoint["revision"] == official_download["revision"],
            "Checkpoint repository or revision mismatch")
    checked["checkpoint"] = checkpoint
    return checked


def audit(output):
    output = Path(output).expanduser().resolve()
    manifest = read_json(output / "run_manifest.json")
    summary = read_json(output / "summary.json")
    completion = read_json(output / "completion.json")
    progress = read_json(output / "progress.json")
    require(summary["status"] == completion["status"] == progress["status"] == "complete",
            "Inference has not completed successfully")
    require(summary == completion == progress, "Final completion records differ")
    require(manifest["kind"] == "node21_official_radzero_zero_shot_v1", "Unexpected run kind")
    payload = {key: value for key, value in manifest.items()
               if key not in ("fingerprint", "created_at", "score_description", "map_description")}
    require(fingerprint(payload) == manifest["fingerprint"] == summary["fingerprint"],
            "Run identity fingerprint mismatch")
    require(manifest["gpus"] == summary["gpus"] == [0, 1],
            "This audit expects physical GPUs 0 and 1")
    require(manifest["class_names"] == ["Nodule"]
            and manifest["prompts"] == ["There is Lung Nodule"],
            "Unexpected class or prompt")
    require(manifest["precision"] == "float32, TF32 disabled", "Unexpected precision")
    checked_sources = source_hash_checks(manifest)
    require(sha256(MODEL_DIR / "config.json") == manifest["model_config_sha256"]
            and sha256(MODEL_DIR / "preprocessor_config.json") == manifest["preprocessor_sha256"],
            "Model or image processor configuration changed")

    metadata = manifest["metadata"]
    require(metadata["simulated_metadata_used"] is False, "Simulated targets must not be used")
    table = Path(metadata["metadata_csv"])
    require(table.name == "metadata.csv", "Unexpected annotation source")
    labels_by_name, source_counts = metadata_images(table)
    images_root = table.parent / "images"
    require({path.name for path in images_root.glob("*.mha")} == set(labels_by_name),
            "Image inventory does not match the metadata")
    names = sorted(labels_by_name)
    limit = manifest["limit"]
    require(isinstance(limit, int) and limit >= 0, "Invalid evaluation limit")
    if limit:
        names = names[:limit]
    paths = [str((images_root / name).resolve()) for name in names]
    labels = np.asarray([labels_by_name[name] for name in names], dtype=np.int8)
    count = len(paths)
    require(manifest["image_paths"] == paths, "Run image order differs from the source metadata")
    require(np.array_equal(np.asarray(manifest["labels"]).ravel(), labels),
            "Run labels differ from the source metadata")
    require(manifest["image_count"] == summary["total_images"] == count,
            "Run image count differs")
    require(summary["completed_images"] == count, "Not all images completed")
    inventory = []
    for path in paths:
        stat = Path(path).stat()
        inventory.append([path, stat.st_size, stat.st_mtime_ns])
    require(fingerprint(inventory) == manifest["image_inventory_sha256"],
            "Image inventory changed after inference")
    require(metadata["full_unique_images"] == source_counts["images"]
            and metadata["full_positive_images"] == source_counts["positive_images"]
            and metadata["full_negative_images"] == source_counts["negative_images"]
            and metadata["full_box_rows"] == source_counts["positive_boxes"]
            and metadata["source_rows"] == source_counts["annotation_rows"],
            "Recorded source statistics differ")

    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == count, "Prediction count differs")
    require([int(row["index"]) for row in rows] == list(range(count)),
            "Prediction indices contain duplicates, gaps or an unexpected order")
    require([row["image_path"] for row in rows] == paths, "Prediction image paths/order differ")
    require(len({row["image_path"] for row in rows}) == count, "Repeated prediction images")
    require(np.array_equal([int(row["label:Nodule"]) for row in rows], labels),
            "Prediction labels differ")
    logits = np.asarray([float(row["logit:Nodule"]) for row in rows])
    scores = np.asarray([float(row["score:Nodule"]) for row in rows])
    require(np.isfinite(scores).all(), "Nonfinite scores")
    np.testing.assert_allclose(scores, expit(logits), atol=1e-12, rtol=1e-12)
    values = metric_values(labels, logits)
    check_metrics(read_json(output / "metrics.json"), values, "Nodule")
    check_metrics(summary["metrics"], values, "Nodule")

    batch_records, offset = [], 0
    for path in sorted((output / "batches").glob("batch_*.npz")):
        parts = path.stem.split("_")
        require(len(parts) == 3, f"Unexpected batch filename: {path}")
        first, stop = int(parts[1]), int(parts[2])
        require(first == offset and first < stop <= count, f"Batch coverage gap or overlap: {path}")
        length = stop - first
        with np.load(path, allow_pickle=False) as batch:
            require(np.array_equal(batch["indices"], np.arange(first, stop)),
                    f"Batch indices differ: {path}")
            for key, shape in {"logits": (length, 1), "similarity_scores": (length, 1, 37, 37),
                               "image_hw": (length, 2)}.items():
                require(batch[key].shape == shape and np.isfinite(batch[key]).all(),
                        f"Invalid {key} in {path}")
            require(batch["logits"].dtype == batch["similarity_scores"].dtype == np.float32,
                    f"Expected FP32 predictions/maps: {path}")
            require((batch["image_hw"] > 0).all(), f"Invalid image dimensions: {path}")
            require(np.array_equal(batch["logits"].ravel().astype(np.float64), logits[first:stop]),
                    f"Saved batch and exported CSV logits differ: {path}")
        batch_records.append({"name": path.name, "images": length, "sha256": sha256(path)})
        offset = stop
    require(offset == count, "Saved batches do not cover all predictions")
    worker_checks = {}
    for rank, gpu in enumerate(manifest["gpus"]):
        directory = output / "workers" / f"gpu{gpu}"
        state = read_json(directory / "progress.json")
        require(state == read_json(directory / "completion.json")
                == summary["workers"][str(gpu)], f"GPU {gpu} completion records differ")
        start, end = count * rank // 2, count * (rank + 1) // 2
        require(manifest["shards"][rank] == {"start": start, "end": end},
                f"Unexpected GPU {gpu} shard")
        require(state["status"] == "complete" and state["rank"] == rank
                and state["physical_gpu"] == gpu and state["cuda_visible_devices"] == str(gpu)
                and state["start"] == start and state["end"] == end
                and state["total_images"] == state["completed_images"] == end - start,
                f"GPU {gpu} did not complete its expected shard")
        reference = read_json(directory / "reference_equivalence.json")
        require(reference["passed"] is True and reference["checkpoint"] == manifest["checkpoint"],
                f"GPU {gpu} official-computation reference check missing or mismatched")
        checks = reference["checks"]
        expected_sizes = {1, min(2, manifest["batch_size"], end - start)}
        require({entry["batch_size"] for entry in checks} == expected_sizes,
                f"GPU {gpu} reference check sizes differ")
        for entry in checks:
            require(entry["classes"] == 1, f"GPU {gpu} reference class count differs")
            for key in ("max_logit_error", "max_map_error"):
                require(math.isfinite(entry[key]) and entry[key] >= 0,
                        f"GPU {gpu} invalid reference error")
        worker_checks[str(gpu)] = {"images": end - start, "reference_checks": checks}

    return {
        "status": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output), "audit_code_sha256": sha256(__file__),
        "run_fingerprint": manifest["fingerprint"],
        "run_manifest_sha256": sha256(output / "run_manifest.json"),
        "predictions_sha256": sha256(output / "predictions.csv"),
        "metrics_sha256": sha256(output / "metrics.json"),
        "full_public_collection": count == source_counts["images"],
        "source_counts": source_counts, "recomputed_metrics": values,
        "workers": worker_checks, "batches": batch_records, "checked_sources": checked_sources,
        "image_inventory_check": "path, size and mtime_ns; not a cryptographic pixel-content audit",
        "scope": ("Image-level nodule classification on public NODE21 training-source data. "
                  "This is not private challenge testing or a detection/localization metric."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed inference output directory")
    args = parser.parse_args()
    target = args.output.expanduser().resolve() / "independent_audit.json"
    try:
        result = audit(args.output)
    except Exception as error:
        result = {"status": "failed", "error": repr(error),
                  "audited_at": datetime.now(timezone.utc).isoformat(),
                  "traceback": traceback.format_exc()}
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: result[key] for key in
                      ("status", "output", "full_public_collection", "recomputed_metrics", "workers")},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
