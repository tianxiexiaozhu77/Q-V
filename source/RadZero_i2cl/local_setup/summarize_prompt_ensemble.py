#!/usr/bin/env python3
"""Audit fixed prompt-ensemble test predictions and compare the saved baseline.

This CPU-only program never loads a model or selects prompts using test scores.
It writes summaries only after every requested completed dataset passes audit.
Use --limit N only for a separately saved smoke run with exactly N rows per set.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from batch_data import DATASETS, load_dataset
from infer import ASSETS_DIR


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_close(actual, expected, name, atol=1e-12, rtol=1e-12):
    if actual is None or expected is None:
        require(actual is expected, f"{name}: undefined metric differs")
    else:
        require(
            bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            f"{name}: {actual!r} differs from expected {expected!r}",
        )


def predictions(path, spec, count):
    """Check all rows, column order, labels and score transforms before use."""
    names = spec["class_names"]
    expected_columns = (
        ["index", "image_path"]
        + [f"label:{name}" for name in names]
        + [f"logit:{name}" for name in names]
        + [f"score:{name}" for name in names]
    )
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == expected_columns, f"{path}: column order differs")
        for index, row in enumerate(reader):
            require(index < count, f"{path}: more than {count} prediction rows")
            require(int(row["index"]) == index, f"{path}: index mismatch at row {index}")
            require(
                row["image_path"] == spec["image_paths"][index],
                f"{path}: image order mismatch at row {index}",
            )
            labels = np.asarray([int(row[f"label:{name}"]) for name in names])
            require(
                np.array_equal(labels, spec["labels"][index]),
                f"{path}: labels mismatch at row {index}",
            )
            logits = np.asarray([float(row[f"logit:{name}"]) for name in names])
            scores = np.asarray([float(row[f"score:{name}"]) for name in names])
            require(
                bool(np.isfinite(logits).all() and np.isfinite(scores).all()),
                f"{path}: non-finite output at row {index}",
            )
            # Clipping at the float64 exp range preserves the sigmoid to precision.
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))
            require(
                bool(np.allclose(scores, sigmoid, atol=1e-12, rtol=1e-12)),
                f"{path}: score is not sigmoid(logit) at row {index}",
            )
            values.append(logits)
    require(len(values) == count, f"{path}: {len(values)} rows, expected {count}")
    return np.asarray(values, dtype=np.float64)


def recompute_metrics(spec, logits):
    """Recompute directly from checked labels/CSV, without runner metric code."""
    labels = np.asarray(spec["labels"], dtype=np.int8)
    require(labels.shape == logits.shape, "Metrics labels/logits shape differs")

    def binary(y, score):
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        roc = float(roc_auc_score(y, score)) if positives and negatives else None
        if positives:
            precision, recall, _ = precision_recall_curve(y, score)
            pr = float(auc(recall, precision))
        else:
            pr = None
        return {"positives": positives, "negatives": negatives, "auroc": roc, "auprc": pr}

    per_class = {
        name: binary(labels[:, column], logits[:, column])
        for column, name in enumerate(spec["class_names"])
    }
    result = {
        "dataset": spec["name"],
        "sample_count": int(len(labels)),
        "class_count": int(labels.shape[1]),
        "per_class": per_class,
    }
    micro = binary(labels.ravel(), logits.ravel())
    for metric in ("auroc", "auprc"):
        defined = [row[metric] for row in per_class.values() if row[metric] is not None]
        result[f"macro_{metric}"] = float(np.mean(defined)) if defined else None
        result[f"macro_{metric}_defined_classes"] = len(defined)
        result[f"micro_{metric}"] = micro[metric]
    if spec["name"] == "OpenI":
        frequency = labels.sum(axis=0)
        threshold = float(np.percentile(frequency, 65))
        masks = {
            "head": frequency > threshold,
            "medium": (frequency <= threshold) & (frequency > 10),
            "tail": frequency <= 10,
        }
        groups = {"head_threshold_count": threshold}
        for group, mask in masks.items():
            names = [spec["class_names"][i] for i in np.flatnonzero(mask)]
            defined = [per_class[name]["auroc"] for name in names if per_class[name]["auroc"] is not None]
            groups[group] = {
                "class_names": names,
                "class_count": len(names),
                "defined_auroc_class_count": len(defined),
                "macro_auroc": float(np.mean(defined)) if defined else None,
            }
        result["openi_groups"] = groups
    return result


def verify_metrics(saved, recomputed, where):
    """Every recomputed field must agree; descriptive runner fields may differ."""
    for key, expected in recomputed.items():
        require(key in saved, f"{where}: missing {key}")
        actual = saved[key]
        if isinstance(expected, dict):
            require(isinstance(actual, dict), f"{where}.{key}: expected object")
            verify_metrics(actual, expected, f"{where}.{key}")
        elif isinstance(expected, float) or expected is None:
            check_close(actual, expected, f"{where}.{key}")
        else:
            require(actual == expected, f"{where}.{key}: differs")


def array_header(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(f"{member}.npy") as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
    return shape, fortran, dtype


def check_manifest(manifest, spec, count, source_hashes, where):
    require(manifest.get("dataset") == spec["name"], f"{where}: dataset differs")
    require(manifest.get("image_count") == count, f"{where}: image_count differs")
    require(
        manifest.get("full_test_count") == len(spec["image_paths"]),
        f"{where}: full_test_count differs",
    )
    require(manifest.get("class_names") == spec["class_names"], f"{where}: classes differ")
    require(manifest.get("prompts") == spec["prompts"], f"{where}: author prompts differ")
    fingerprint = manifest.get("fingerprint", "")
    require(
        isinstance(fingerprint, str) and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint),
        f"{where}: missing/invalid fingerprint",
    )
    require(manifest.get("source_sha256") == source_hashes, f"{where}: source hashes differ")


def check_progress(path, count):
    progress = read_json(path)
    require(progress.get("status") == "complete", f"{path}: dataset is not complete")
    require(progress.get("completed_images") == count, f"{path}: completed count differs")
    require(progress.get("total_images") == count, f"{path}: total count differs")
    return progress


def audit_dataset(name, args):
    spec = load_dataset(name, ASSETS_DIR / "data")
    full_count = len(spec["image_paths"])
    count = min(args.limit, full_count) if args.limit else full_count
    classes = len(spec["class_names"])
    directory, baseline_directory = args.run_dir / name, args.baseline_dir / name
    manifest_path, baseline_manifest_path = (
        directory / "manifest.json",
        baseline_directory / "manifest.json",
    )
    manifest, baseline_manifest = read_json(manifest_path), read_json(baseline_manifest_path)
    source_hashes = {str(path): sha256(Path(path)) for path in spec["source_files"]}
    check_manifest(manifest, spec, count, source_hashes, str(manifest_path))
    check_manifest(baseline_manifest, spec, full_count, source_hashes, str(baseline_manifest_path))
    progress = check_progress(directory / "progress.json", count)
    check_progress(baseline_directory / "progress.json", full_count)
    config = manifest.get("prompt_ensemble")
    require(isinstance(config, dict) and bool(config), f"{name}: missing prompt_ensemble config")
    from prompt_ensemble import build_prompt_config
    require(
        config == build_prompt_config(spec),
        f"{name}: configuration differs from the fixed five-template protocol",
    )
    csv_path = directory / "predictions.csv"
    baseline_csv_path = baseline_directory / "predictions.csv"
    logits = predictions(csv_path, spec, count)
    baseline_full_logits = predictions(baseline_csv_path, spec, full_count)
    baseline_logits = baseline_full_logits[:count]
    selected = dict(spec, image_paths=spec["image_paths"][:count], labels=spec["labels"][:count])
    metrics = recompute_metrics(selected, logits)
    baseline_metrics = recompute_metrics(selected, baseline_logits)
    verify_metrics(read_json(directory / "metrics.json"), metrics, f"{name} ensemble metrics")
    if "metrics" in progress:
        verify_metrics(progress["metrics"], metrics, f"{name} progress metrics")
    full_baseline_metrics = baseline_metrics if count == full_count else recompute_metrics(spec, baseline_full_logits)
    verify_metrics(
        read_json(baseline_directory / "metrics.json"),
        full_baseline_metrics,
        f"{name} baseline metrics",
    )
    chunks = sorted((directory / "batches").glob("batch_*.npz"))
    require(bool(chunks), f"{name}: missing NPZ batches")
    offset, max_baseline_error, max_mean_error, map_payload_count = 0, 0.0, 0.0, 0
    batch_hashes = {}
    for chunk_index, path in enumerate(chunks):
        with np.load(path, allow_pickle=False) as chunk:
            indices = chunk["indices"]
            size = len(indices)
            require(size > 0 and offset + size <= count, f"{path}: invalid batch length")
            require(
                indices.dtype.kind in "iu" and np.array_equal(indices, np.arange(offset, offset + size)),
                f"{path}: missing, duplicated or unordered indices",
            )
            batch_logits, template_logits = chunk["logits"], chunk["template_logits"]
            require(batch_logits.shape == (size, classes), f"{path}: logits shape differs")
            require(template_logits.shape == (size, classes, 5), f"{path}: template_logits shape differs")
            require(
                batch_logits.dtype == np.dtype("float32") and template_logits.dtype == np.dtype("float32"),
                f"{path}: expected float32 logits",
            )
            require(
                bool(np.isfinite(batch_logits).all() and np.isfinite(template_logits).all()),
                f"{path}: non-finite logits",
            )
            require(
                np.array_equal(batch_logits, logits[offset:offset + size]),
                f"{path}: NPZ and CSV logits differ",
            )
            averaged = template_logits.mean(axis=-1)
            check_close(batch_logits, averaged, f"{path}: template mean", atol=2e-5, rtol=1e-4)
            first_template = template_logits[:, :, 0]
            expected = baseline_logits[offset:offset + size]
            check_close(
                first_template, expected, f"{path}: original-prompt baseline",
                atol=args.atol, rtol=args.rtol,
            )
            max_baseline_error = max(max_baseline_error, float(np.abs(first_template - expected).max()))
            max_mean_error = max(max_mean_error, float(np.abs(batch_logits - averaged).max()))
            sizes = chunk["image_hw"]
            require(
                sizes.shape == (size, 2) and sizes.dtype.kind in "iu" and bool((sizes > 0).all()),
                f"{path}: invalid original image dimensions",
            )
            shape, fortran, dtype = array_header(path, "similarity_scores")
            require(
                shape == (size, classes, 37, 37) and dtype == np.dtype("float32") and not fortran,
                f"{path}: map header differs",
            )
            if chunk_index in (0, len(chunks) - 1):
                require(bool(np.isfinite(chunk["similarity_scores"]).all()), f"{path}: non-finite maps")
                map_payload_count += 1
            # Hashing the compressed file records the exact artifact without
            # repeatedly decompressing all spatial-map payloads.
            batch_hashes[path.name] = sha256(path)
            offset += size
    require(offset == count, f"{name}: NPZ batches cover {offset}, expected {count}")
    run_progress = read_json(args.run_dir / "progress.json")
    gpus = run_progress.get("gpus")
    require(isinstance(gpus, list) and bool(gpus), "Run progress must list the physical GPUs")
    references = {}
    for gpu in gpus:
        reference_path = args.run_dir / "workers" / f"gpu{gpu}" / f"{name}_reference_equivalence.json"
        reference = read_json(reference_path)
        require(reference.get("passed") is True, f"{reference_path}: reference check missing/failed")
        require(bool(reference.get("checks")), f"{reference_path}: reference checks are empty")
        references[f"gpu{gpu}"] = reference
    report = {
        "passed": True,
        "images": count,
        "full_test_count": full_count,
        "classes": classes,
        "chunks": len(chunks),
        "fingerprint": manifest["fingerprint"],
        "baseline_fingerprint": baseline_manifest["fingerprint"],
        "prompt_ensemble": config,
        "prompt_ensemble_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "csv_order_labels_and_prompts_match_author_and_baseline": True,
        "csv_scores_match_sigmoid_logits": True,
        "npz_indices_complete_and_contiguous": True,
        "npz_logits_exactly_match_csv": True,
        "ensemble_matches_equal_mean_of_five_template_logits": True,
        "max_ensemble_mean_abs_error": max_mean_error,
        "original_prompt_matches_previous_baseline": True,
        "max_original_prompt_baseline_abs_error": max_baseline_error,
        "original_prompt_tolerances": {"atol": args.atol, "rtol": args.rtol},
        "map_header_checked_all_chunks": True,
        "map_payload_finite_rechecked_chunks": map_payload_count,
        "map_validation_scope": "All map headers; first and last chunk payloads. Runner checks every generated map for finite values.",
        "metrics_recomputed_from_checked_csv_and_author_labels": True,
        "artifact_sha256": {
            "manifest.json": sha256(manifest_path),
            "predictions.csv": sha256(csv_path),
            "metrics.json": sha256(directory / "metrics.json"),
            "baseline_manifest.json": sha256(baseline_manifest_path),
            "baseline_predictions.csv": sha256(baseline_csv_path),
        },
        "batch_sha256": batch_hashes,
        "source_sha256": source_hashes,
        "reference_equivalence": references,
        "validated_at": now(),
    }
    return report, baseline_metrics, metrics


def delta_points(ensemble, baseline):
    return None if ensemble is None or baseline is None else (ensemble - baseline) * 100.0


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def timing_metadata(directory):
    """Keep provenance from workers without assuming their exact directory name."""
    result = {}
    candidates = [
        directory / "progress.json", directory / "run_config.json",
        directory / "run_manifest.json", directory / "launch.json", directory / "timing.json",
    ]
    candidates.extend(directory.glob("workers/*/progress.json"))
    candidates.extend(directory.glob("workers/*/summary.json"))
    candidates.extend(directory.glob("workers/*.json"))
    candidates.extend(directory.glob("worker_*/progress.json"))
    candidates.extend(directory.glob("gpu*/progress.json"))
    for path in sorted(set(candidates)):
        if not path.is_file():
            continue
        value = read_json(path)
        selected = {
            key: item for key, item in value.items()
            if key in {
                "status", "started_at", "completed_at", "finished_at", "updated_at",
                "elapsed_seconds", "elapsed_seconds_this_session", "device", "gpu",
                "gpu_name", "physical_gpu", "cuda_visible_devices", "pid",
                "workers", "batch_size", "checkpoint", "precision", "limit",
                "model_config_sha256", "script_sha256", "gpus",
            }
        }
        result[str(path.relative_to(directory))] = selected
    starts, ends = [], []
    for value in result.values():
        if value.get("started_at"):
            starts.append(datetime.fromisoformat(value["started_at"].replace("Z", "+00:00")))
        for key in ("completed_at", "finished_at"):
            if value.get(key):
                ends.append(datetime.fromisoformat(value[key].replace("Z", "+00:00")))
                break
    if starts and ends:
        result["observed_worker_wall_time"] = {
            "started_at": min(starts).isoformat(),
            "completed_at": max(ends).isoformat(),
            "elapsed_seconds": (max(ends) - min(starts)).total_seconds(),
            "definition": "Earliest recorded start to latest recorded completion; worker durations are not summed.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ASSETS_DIR / "outputs/public5_prompt_ensemble_gpu01")
    parser.add_argument("--baseline-dir", type=Path, default=ASSETS_DIR / "outputs/public5_gpu0")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int, default=0, help="Exact smoke-run prefix length; 0 requires full tests")
    parser.add_argument("--atol", type=float, default=2e-5, help="Original-prompt vs previous-baseline absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Original-prompt vs previous-baseline relative tolerance")
    args = parser.parse_args()
    if args.limit < 0 or args.atol < 0 or args.rtol < 0:
        parser.error("Limit and numerical tolerances must be nonnegative")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    args.run_dir = args.run_dir.expanduser().resolve()
    args.baseline_dir = args.baseline_dir.expanduser().resolve()
    if args.run_dir == args.baseline_dir:
        parser.error("Ensemble run and baseline directories must differ")
    if not args.run_dir.is_dir():
        parser.error(f"Run directory does not exist: {args.run_dir}")
    validation = {
        "status": "running",
        "mode": "smoke" if args.limit else "full_test",
        "run_directory": str(args.run_dir),
        "baseline_directory": str(args.baseline_dir),
        "requested_datasets": args.datasets,
        "limit_per_dataset": args.limit,
        "validation_device": "CPU",
        "datasets": {},
        "errors": [],
        "started_at": now(),
    }
    comparisons, per_class, summaries = [], [], {}
    try:
        for name in args.datasets:
            report, baseline, ensemble = audit_dataset(name, args)
            validation["datasets"][name] = report
            row = {"dataset": name, "images": ensemble["sample_count"], "classes": ensemble["class_count"]}
            for metric in ("auroc", "auprc"):
                row[f"baseline_macro_{metric}"] = baseline[f"macro_{metric}"]
                row[f"ensemble_macro_{metric}"] = ensemble[f"macro_{metric}"]
                row[f"delta_macro_{metric}_percentage_points"] = delta_points(
                    ensemble[f"macro_{metric}"], baseline[f"macro_{metric}"],
                )
            comparisons.append(row)
            for class_name, metrics in ensemble["per_class"].items():
                entry = {
                    "dataset": name, "class": class_name,
                    "images": ensemble["sample_count"],
                    "positives": metrics["positives"], "negatives": metrics["negatives"],
                }
                for metric in ("auroc", "auprc"):
                    entry[f"baseline_{metric}"] = baseline["per_class"][class_name][metric]
                    entry[f"ensemble_{metric}"] = metrics[metric]
                    entry[f"delta_{metric}_percentage_points"] = delta_points(
                        metrics[metric], baseline["per_class"][class_name][metric],
                    )
                per_class.append(entry)
            summaries[name] = {
                "status": "complete",
                "completed_images": ensemble["sample_count"],
                "total_images": ensemble["sample_count"],
                "full_test_count": report["full_test_count"],
                "baseline_metrics": baseline,
                "metrics": ensemble,
                "prompt_ensemble": report["prompt_ensemble"],
                "prompt_ensemble_sha256": report["prompt_ensemble_sha256"],
                "fingerprint": report["fingerprint"],
                "baseline_fingerprint": report["baseline_fingerprint"],
                "comparison": row,
            }
            print(json.dumps({"dataset": name, "validation": "passed", **row}), flush=True)
        validation.update({
            "status": "passed", "validated_at": now(),
            "total_images": sum(row["images"] for row in comparisons),
            "dataset_count": len(comparisons),
        })
        summary = {
            "status": "smoke_complete" if args.limit else "complete",
            "mode": validation["mode"],
            "validated_at": validation["validated_at"],
            "run_directory": str(args.run_dir),
            "baseline_directory": str(args.baseline_dir),
            "limit_per_dataset": args.limit,
            "total_images": validation["total_images"],
            "dataset_count": len(comparisons),
            "all_five_full_test_sets_complete": not args.limit and set(args.datasets) == set(DATASETS),
            "method": "Five fixed positive prompts per class; equal mean of raw logits before sigmoid.",
            "selection_policy": "Fixed templates and equal weights; no best-template selection or test-set tuning.",
            "scope": "Image-level classification only; no training, grounding or segmentation evaluation.",
            "metric_units": "AUROC/AUPRC are 0..1; differences are percentage points.",
            "auprc_method": "Trapezoidal auc(recall, precision), not average precision.",
            "datasets": summaries,
            "timing": timing_metadata(args.run_dir),
            "audit_script_sha256": sha256(Path(__file__)),
        }
        write_csv(args.run_dir / "comparison.csv", comparisons)
        write_csv(args.run_dir / "per_class_comparison.csv", per_class)
        write_json(args.run_dir / "validation.json", validation)
        write_json(args.run_dir / "summary.json", summary)
        print(json.dumps({
            "status": summary["status"], "datasets": len(comparisons),
            "images": summary["total_images"], "output": str(args.run_dir),
        }), flush=True)
        return 0
    except Exception as error:
        validation.update({"status": "failed", "validated_at": now()})
        validation["errors"].append(f"{type(error).__name__}: {error}")
        write_json(args.run_dir / "validation.json", validation)
        # Never leave a previous successful summary looking authoritative after
        # a failed re-audit; retain diagnostics while marking the summary failed.
        write_json(args.run_dir / "summary.json", {
            "status": "validation_failed", "mode": validation["mode"],
            "validated_at": validation["validated_at"], "errors": validation["errors"],
            "all_five_full_test_sets_complete": False,
        })
        print(validation["errors"][-1], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
