#!/usr/bin/env python3
"""Support-only score residuals on frozen RadZero features.

These are cache/prototype/ridge score-residual experiments, not the original
I2CL algorithm. Unmodified RadZero maps do not explain the modified scores.
All choices are fixed by leave-one-support-group-out predictions before test
labels are loaded. Repeated experiment-family development against this test
set nevertheless makes the overall overnight exercise exploratory.
"""
from __future__ import annotations

import os

# Set before importing NumPy/scikit-learn, and also enforce at runtime.
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "4"

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics


VERSION = "radzero-frozen-support-score-residual-v1"
FEATURES = ("class_local", "dino_cls", "aligned_cls", "dino_mean", "aligned_mean")


class DeadlineReached(RuntimeError):
    pass


def parse_deadline(value):
    if value is None:
        return None
    try:
        result = float(value)
    except ValueError:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("--deadline must include a timezone")
        result = instant.timestamp()
    if not math.isfinite(result):
        raise ValueError("Deadline must be finite")
    return result


def check_deadline(deadline):
    if deadline is not None and time.time() >= deadline:
        raise DeadlineReached("Requested deadline reached; no further work started")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


def array_digest(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def normalized_features(values, classes):
    """Per-image normalization, with no fitted or test-population statistics."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (2, 3) or not len(values):
        raise ValueError("Features must be nonempty [N,D] or [N,C,D]")
    if values.ndim == 3 and values.shape[1] != classes:
        raise ValueError("Class-conditioned feature columns differ from labels")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite features")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError("Zero-norm feature")
    return values / norms


def class_feature(features, class_index):
    return features[:, class_index] if features.ndim == 3 else features


def sigmoid(values):
    # Avoid overflow without modifying the actual logits used for metrics.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def fit_residual(features, labels, baseline_logits, method, parameter):
    """Fit only support arrays. No test objects/labels are accepted."""
    labels = np.asarray(labels)
    baseline_logits = np.asarray(baseline_logits, dtype=np.float64)
    if labels.ndim != 2 or len(features) != len(labels):
        raise ValueError("Support feature/label shapes differ")
    if baseline_logits.shape != labels.shape or not np.isfinite(baseline_logits).all():
        raise ValueError("Support baseline logits are invalid")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Support labels must be binary")
    if method not in ("cache", "proto", "ridge"):
        raise ValueError(f"Unknown method {method}")
    if method in ("cache", "ridge") and (not math.isfinite(parameter) or parameter <= 0):
        raise ValueError("Kernel temperature/ridge penalty must be positive")
    classes = labels.shape[1]
    dimension = features.shape[-1]
    positives = labels.sum(axis=0).astype(int)
    negatives = len(labels) - positives
    valid = (positives > 0) & (negatives > 0)
    state = {
        "method": method, "parameter": float(parameter),
        "valid": valid, "positive_counts": positives, "negative_counts": negatives,
        "support_count": len(labels), "class_count": classes,
        "features": np.asarray(features), "labels": labels.astype(np.int8),
    }
    directions = np.zeros((classes, dimension), dtype=np.float64)
    dual = np.zeros((len(labels), classes), dtype=np.float64)
    for c in range(classes):
        if not valid[c]:
            continue
        x = class_feature(features, c)
        positive = labels[:, c] == 1
        if method == "proto":
            directions[c] = x[positive].mean(axis=0) - x[~positive].mean(axis=0)
        elif method == "ridge":
            # Regularized baseline-probability residual, in dual form.
            # Class balancing is estimated from this fit subset, not full
            # support during OOF. Normalize weights to mean 1 for stable L2.
            weights = np.where(positive, negatives[c] / positives[c], 1.0)
            weights /= weights.mean()
            root_weights = np.sqrt(weights)
            target = labels[:, c] - sigmoid(baseline_logits[:, c])
            gram = x @ x.T
            system = gram * np.outer(root_weights, root_weights)
            system.flat[::len(system) + 1] += parameter
            dual[:, c] = root_weights * np.linalg.solve(system, root_weights * target)
            directions[c] = x.T @ dual[:, c]
    state["directions"] = directions
    state["dual"] = dual
    return state


def predict_residual(state, query_features):
    """Predict without labels. Return zero for a class absent in fit labels."""
    result = np.zeros((len(query_features), state["class_count"]), dtype=np.float64)
    for c in range(state["class_count"]):
        if not state["valid"][c]:
            continue
        query = class_feature(query_features, c)
        if state["method"] in ("proto", "ridge"):
            result[:, c] = query @ state["directions"][c]
        else:
            source = class_feature(state["features"], c)
            similarity = np.clip(query @ source.T, -1.0, 1.0)
            kernel = np.exp(-state["parameter"] * (1.0 - similarity))
            positive = state["labels"][:, c] == 1
            result[:, c] = kernel[:, positive].mean(axis=1) - kernel[:, ~positive].mean(axis=1)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite residual prediction")
    return result


def leave_group_out(features, labels, baseline_logits, groups, method, parameter,
                    deadline=None):
    """Rebuild all state after removing the held-out image's entire group."""
    groups = np.asarray(groups, dtype=str)
    if groups.shape != (len(labels),) or np.any(groups == ""):
        raise ValueError("Need one nonempty group per support row")
    if len(np.unique(groups)) < 2:
        raise ValueError("OOF requires at least two support groups")
    output = np.zeros_like(baseline_logits, dtype=np.float64)
    missing_rows = np.zeros(labels.shape[1], dtype=np.int64)
    folds = []
    for group in np.unique(groups):
        check_deadline(deadline)
        held = groups == group
        kept = ~held
        state = fit_residual(features[kept], labels[kept], baseline_logits[kept],
                             method, parameter)
        output[held] = predict_residual(state, features[held])
        missing = np.flatnonzero(~state["valid"]).tolist()
        missing_rows += (~state["valid"]).astype(np.int64) * int(held.sum())
        folds.append({
            "held_group_sha256": hashlib.sha256(group.encode()).hexdigest(),
            "held_indices": np.flatnonzero(held).tolist(),
            "fit_indices": np.flatnonzero(kept).tolist(),
            "fit_positive_counts": state["positive_counts"].tolist(),
            "fit_negative_counts": state["negative_counts"].tolist(),
            "zero_residual_classes": missing,
        })
    return output, {"fold_count": len(folds), "folds": folds,
                    "missing_state_rows_per_class": missing_rows.tolist()}


def metric_score(metrics):
    values = [metrics["macro_auroc"], metrics["macro_auprc"]]
    return sum(value for value in values if value is not None) / max(
        1, sum(value is not None for value in values))


def choose_candidates(records, top_k):
    """One best OOF candidate per (method,feature), then overall top K."""
    best = {}
    for record in records:
        config = record["config"]
        key = (config["method"], config["feature"])
        ordering = (record["selection_score"], -config["strength"])
        if key not in best or ordering > best[key][0]:
            best[key] = (ordering, record)
    ranked = sorted((value[1] for value in best.values()),
                    key=lambda row: (-row["selection_score"],
                                     row["config"]["strength"], row["candidate_id"]))
    return ranked[:top_k]


def metric_delta(base, adapted):
    return {key: None if base[key] is None or adapted[key] is None else adapted[key] - base[key]
            for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")}


def evaluate_test_only(split_dir, dataset, logits, baseline_logits, expected_identity,
                       output, method=None):
    """The sole test-label reader; called only after support selection is saved."""
    from i2cl_data import load_spec
    spec = load_spec(split_dir, dataset, "test")
    if sha256(Path(split_dir) / dataset / "test.csv") != expected_identity["csv_sha256"]:
        raise ValueError("Test split checksum differs from frozen-feature cache")
    if list(spec["image_paths"]) != expected_identity["image_paths"]:
        raise ValueError("Test feature image ordering differs from verified split")
    base_metrics = compute_metrics(spec, baseline_logits)
    adapted_metrics = compute_metrics(spec, logits)
    score_types = {
        "proto": "frozen RadZero baseline plus positive-negative support prototype residual",
        "cache": "frozen RadZero baseline plus Tip-Adapter-like exponential support cache residual",
        "ridge": "frozen RadZero baseline plus support-trained frozen-feature score residual",
    }
    adapted_metrics["score_type"] = score_types.get(
        str(method), "frozen RadZero baseline plus support-trained score residual")
    with (Path(output) / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path",
                         *[f"label:{name}" for name in spec["class_names"]],
                         *[f"logit:{name}" for name in spec["class_names"]]])
        for i, (path, labels, row) in enumerate(zip(spec["image_paths"], spec["labels"], logits)):
            writer.writerow([i, path, *labels.tolist(), *row.tolist()])
    result = {"baseline_metrics": base_metrics, "metrics": adapted_metrics,
              "delta": metric_delta(base_metrics, adapted_metrics)}
    write_json(Path(output) / "metrics.json", result)
    return result


def save_fit(path, state, support_identity):
    np.savez(path, method=np.asarray(state["method"]), parameter=state["parameter"],
             valid=state["valid"], positive_counts=state["positive_counts"],
             negative_counts=state["negative_counts"], support_features=state["features"],
             support_labels=state["labels"], directions=state["directions"],
             dual=state["dual"], support_identity=np.asarray(support_identity))


def cache_view(cache_path):
    """Read the extractor's published manifest; schema finalized with producer."""
    cache_path = Path(cache_path).resolve()
    manifest_path = cache_path / "manifest.json" if cache_path.is_dir() else cache_path
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Frozen-feature cache is not complete")
    return manifest_path, manifest


def run(args):
    # The producer schema adapter is deliberately isolated here.
    manifest_path, manifest = cache_view(args.cache)
    cache_dir = manifest_path.parent
    dataset = manifest["dataset"]
    split_dir = Path(manifest["split_dir"])
    if args.dataset and args.dataset != dataset:
        raise ValueError("Dataset differs from feature-cache manifest")
    from i2cl_data import load_spec
    support = load_spec(split_dir, dataset, "support")
    labels = np.asarray(support["labels"], dtype=np.int8)
    classes = len(support["class_names"])
    arrays = manifest["splits"]

    def load_array(split, key):
        feature_keys = {
            "baseline_logits": ("base_logits", None),
            "class_local": ("class_local", None),
            "dino_cls": ("global_dino", 0), "aligned_cls": ("global_aligned", 0),
            "dino_mean": ("global_dino", 1), "aligned_mean": ("global_aligned", 1),
        }
        array_key, slice_index = feature_keys[key]
        record = arrays[split]["files"][array_key]
        path = cache_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Cache array checksum mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"Cache array shape/dtype mismatch: {path}")
        if slice_index is not None:
            if array.ndim != 3 or array.shape[1] != 2:
                raise ValueError("Expected global features [N,2,D], CLS then patch mean")
            return array[:, slice_index]
        return array

    identities = {}
    for split in ("support", "test"):
        metadata_path = cache_dir / f"{split}_metadata.json"
        if sha256(metadata_path) != manifest["auxiliary_files"][metadata_path.name]["sha256"]:
            raise ValueError("Feature-cache metadata checksum mismatch")
        metadata = json.loads(metadata_path.read_text())
        if fingerprint(metadata) != manifest[f"{split}_identity"]:
            raise ValueError("Feature-cache split metadata identity mismatch")
        if "labels" in metadata:
            raise ValueError("Feature metadata must not include test or support labels")
        identities[split] = {
            **metadata,
            "csv_sha256": manifest["split_sha256"][str(split_dir / dataset / f"{split}.csv")],
        }
    support_identity, test_identity = identities["support"], identities["test"]
    if sha256(split_dir / dataset / "support.csv") != support_identity["csv_sha256"]:
        raise ValueError("Support split checksum differs from cached features")
    if support["image_paths"] != support_identity["image_paths"]:
        raise ValueError("Support image order differs from cached features")
    if list(support["class_names"]) != manifest["class_names"]:
        raise ValueError("Feature class columns differ from verified support")
    if list(support["groups"]) != support_identity["groups"]:
        raise ValueError("Feature-cache support groups differ from verified split")
    label_path = cache_dir / "support_labels.npy"
    if sha256(label_path) != manifest["auxiliary_files"][label_path.name]["sha256"]:
        raise ValueError("Cached support-label checksum mismatch")
    if not np.array_equal(labels, np.load(label_path, allow_pickle=False)):
        raise ValueError("Cached support labels differ from verified split")
    if set(support["groups"]) & set(test_identity["groups"]):
        raise ValueError("Support/test groups overlap")
    if set(support["image_paths"]) & set(test_identity["image_paths"]):
        raise ValueError("Support/test images overlap")
    baseline_support = np.asarray(load_array("support", "baseline_logits"), dtype=np.float64)
    if baseline_support.shape != labels.shape:
        raise ValueError("Support baseline shape mismatch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    write_json(output / "run_config.json", {
        "version": VERSION, "created_at": started, "dataset": dataset,
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": sha256(manifest_path),
        "source_sha256": sha256(__file__), "arguments": vars(args),
        "support_csv_sha256": support_identity["csv_sha256"],
        "test_csv_sha256": test_identity["csv_sha256"],
        "support_count": len(labels), "class_names": support["class_names"],
        "selection": "Mean of macro AUROC/AUPRC on leave-one-support-group-out predictions",
        "test_labels_used_in_fitting_or_selection": False,
        "interpretation": "Exploratory score-residual adaptation; not original I2CL; no adapted maps",
    })
    write_json(output / "status.json", {"status": "support_selection", "started_at": started})
    support_spec = {"name": dataset, "labels": labels, "class_names": support["class_names"]}
    baseline_oof = compute_metrics(support_spec, baseline_support)
    records, feature_cache, fold_audits = [], {}, {}
    try:
        for feature_name in args.features:
            check_deadline(args.deadline)
            feature = normalized_features(load_array("support", feature_name), classes)
            feature_cache[feature_name] = feature
            for method in args.methods:
                parameters = args.taus if method == "cache" else (
                    args.l2s if method == "ridge" else [1.0])
                for parameter in parameters:
                    check_deadline(args.deadline)
                    residual, fold_audit = leave_group_out(
                        feature, labels, baseline_support, support["groups"],
                        method, parameter, args.deadline)
                    key = f"{feature_name}_{method}_{parameter:g}"
                    fold_audits[key] = fold_audit
                    for strength in args.strengths:
                        candidate_id = f"{key}_strength{strength:g}"
                        logits = baseline_support + strength * residual
                        metrics = compute_metrics(support_spec, logits)
                        records.append({
                            "candidate_id": candidate_id,
                            "config": {"method": method, "feature": feature_name,
                                       "parameter": parameter, "strength": strength},
                            "support_oof_metrics": metrics,
                            "support_oof_delta": metric_delta(baseline_oof, metrics),
                            "selection_score": metric_score(metrics),
                            "fold_audit_key": key,
                        })
                    write_json(output / "candidates.json", records)
                    print(f"{dataset}: OOF {key}; {len(records)} candidates", flush=True)
        write_json(output / "fold_audits.json", fold_audits)
        chosen = choose_candidates(records, args.top_k)
        selection = {
            "selected_before_test_labels_loaded": True, "selection_time": time.time(),
            "baseline_support_metrics": baseline_oof,
            "support_labels_sha256": array_digest(labels),
            "support_groups_sha256": hashlib.sha256(
                json.dumps(support["groups"]).encode()).hexdigest(),
            "selection_metric": "mean(macro_AUROC, macro_AUPRC) on support group-OOF",
            "top_k_distinct_method_feature": args.top_k, "candidates": chosen,
        }
        write_json(output / "selection.json", selection)
        selection_hash = sha256(output / "selection.json")
        write_json(output / "status.json", {"status": "selected_test_prediction",
                                           "selection_sha256": selection_hash})
        # First read of held-out features is after OOF choices are immutable.
        baseline_test = np.asarray(load_array("test", "baseline_logits"), dtype=np.float64)
        results = []
        for record in chosen:
            check_deadline(args.deadline)
            config = record["config"]
            destination = output / record["candidate_id"]
            destination.mkdir()
            state = fit_residual(feature_cache[config["feature"]], labels, baseline_support,
                                 config["method"], config["parameter"])
            save_fit(destination / "fit.npz", state, support_identity["csv_sha256"])
            raw_features = load_array("test", config["feature"])
            logits = np.empty_like(baseline_test)
            for first in range(0, len(logits), args.chunk_size):
                check_deadline(args.deadline)
                last = min(len(logits), first + args.chunk_size)
                normalized = normalized_features(raw_features[first:last], classes)
                logits[first:last] = baseline_test[first:last] + config["strength"] * predict_residual(
                    state, normalized)
            np.save(destination / "predictions.npy", logits, allow_pickle=False)
            if sha256(output / "selection.json") != selection_hash:
                raise RuntimeError("Selection changed after reading test features")
            check_deadline(args.deadline)
            evaluation = evaluate_test_only(split_dir, dataset, logits, baseline_test,
                                            test_identity, destination,
                                            method=config["method"])
            result = {**record, **evaluation, "prediction_sha256": sha256(destination / "predictions.npy"),
                      "fit_sha256": sha256(destination / "fit.npz"),
                      "selection_sha256": selection_hash}
            write_json(destination / "result.json", result)
            results.append(result)
            write_json(output / "summary.json", {
                "dataset": dataset, "version": VERSION, "exploratory": True,
                "selection": selection, "results": results,
                "test_labels_used_in_fitting_or_selection": False,
            })
            print(f"{dataset}: {record['candidate_id']} "
                  f"AUROC={evaluation['metrics']['macro_auroc']:.6f} "
                  f"AUPRC={evaluation['metrics']['macro_auprc']:.6f}", flush=True)
        write_json(output / "status.json", {"status": "complete",
                                           "completed_at": datetime.now(timezone.utc).isoformat(),
                                           "selected_count": len(results)})
    except DeadlineReached as error:
        write_json(output / "status.json", {"status": "deadline", "reason": str(error),
                                           "completed_at": datetime.now(timezone.utc).isoformat()})
        raise
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                        choices=("cache", "proto", "ridge"), default=["cache", "proto", "ridge"])
    parser.add_argument("--features", nargs="+", choices=FEATURES,
                        default=["class_local", "dino_cls", "aligned_cls"])
    parser.add_argument("--taus", nargs="+", type=float, default=[1., 5., 10., 20.])
    parser.add_argument("--l2s", nargs="+", type=float, default=[.1, 1., 10.])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0., .1, .3, 1., 3.])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=parse_deadline)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or args.top_k < 1 or args.chunk_size < 1:
        parser.error("threads must be 1..4; top-k and chunk-size must be positive")
    if any(not math.isfinite(x) or x < 0 for x in args.strengths):
        parser.error("strengths must be finite and nonnegative")
    if any(not math.isfinite(x) or x <= 0 for x in args.taus + args.l2s):
        parser.error("taus and l2s must be finite and positive")
    return args


def main():
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        try:
            check_deadline(args.deadline)
            run(args)
        except DeadlineReached as error:
            print(str(error), flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
