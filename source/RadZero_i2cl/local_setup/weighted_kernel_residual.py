#!/usr/bin/env python3
"""Support-only weighted kernel residuals on frozen RadZero class-local features.

Fit y - sigmoid(base_logits) by class-balanced kernel ridge regression. The
original logits remain an offset. Support centering is learned anew inside each
leave-one-group-out fold. All configurations are selected using support OOF
predictions before any test labels are loaded. Overall method development is
exploratory because earlier experiments have already examined these test sets.
This is a score residual, not I2CL; original maps do not explain its correction.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics
from overnight_cache_methods import (
    DeadlineReached, array_digest, cache_view, check_deadline, evaluate_test_only,
    fingerprint, metric_delta, metric_score, parse_deadline, sha256, sigmoid, write_json,
)
from overnight_oof_loss_tiebreak import build_bce_folds, candidate_order, weighted_oof_bce

VERSION = "radzero-support-only-weighted-kernel-residual-v1"
TRANSFORMS = ("normalize", "support_center")
SELECTION_RULE = (
    "One winner per transform/kernel family; descending mean(macro AUROC, macro AUPRC); "
    "exact ties use ascending fold-fit-weighted OOF BCE, then strength and candidate ID"
)


def config_key(config):
    return (f"{config['transform']}_{config['kernel']}"
            f"_tau{config['tau']:g}_l2{config['l2']:g}")


def family_name(config):
    suffix = "raw" if config["transform"] == "normalize" else "centered"
    return f"{config['kernel']}_{suffix}"


def build_grid(args):
    configurations = []
    for transform in args.transforms:
        if "linear" in args.kernels:
            configurations.extend(
                {"transform": transform, "kernel": "linear", "tau": 1., "l2": value}
                for value in args.linear_l2s)
        if "rbf" in args.kernels:
            configurations.extend(
                {"transform": transform, "kernel": "rbf", "tau": tau, "l2": value}
                for tau in args.taus for value in args.rbf_l2s)
    unique = {config_key(value): value for value in configurations}
    return list(unique.values())


def transform_features(features, mean):
    """No query-population statistics; zero centered vectors remain exactly zero."""
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if (values.ndim != 3 or mean.shape != values.shape[1:]
            or not np.isfinite(values).all() or not np.isfinite(mean).all()):
        raise ValueError("Expected finite class-local [N,C,D] features and [C,D] mean")
    centered = values - mean[None]
    norms = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / np.maximum(norms, 1e-12)


def prepare_fit(features, labels, baseline_logits, transform):
    """Accept only fitting support arrays; no validation/test data or labels."""
    raw = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    if (labels.ndim != 2 or not len(labels) or raw.ndim != 3
            or raw.shape[:2] != labels.shape or raw.shape[2] < 1
            or baseline.shape != labels.shape or not np.isfinite(baseline).all()
            or not np.isfinite(raw).all() or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Invalid finite support features, binary labels or baseline logits")
    if transform not in TRANSFORMS:
        raise ValueError("Unknown support feature transform")
    mean = raw.mean(axis=0) if transform == "support_center" else np.zeros(raw.shape[1:])
    features = transform_features(raw, mean)
    positive = labels.sum(axis=0).astype(np.int64)
    negative = len(labels) - positive
    valid = (positive > 0) & (negative > 0)
    weights = np.zeros_like(baseline)
    for c in np.flatnonzero(valid):
        weights[:, c] = np.where(labels[:, c] == 1, negative[c] / positive[c], 1.)
        weights[:, c] /= weights[:, c].mean()
    return {
        "transform": transform, "mean": mean, "features": features,
        "positive_counts": positive, "negative_counts": negative, "valid": valid,
        "weights": weights, "target": labels.astype(float) - sigmoid(baseline),
        "class_count": labels.shape[1], "support_count": len(labels),
        "zero_feature_counts": (np.linalg.norm(features, axis=-1) < 1e-12).sum(axis=0),
    }


def kernel_matrix(first, second, kernel, tau):
    if kernel not in ("linear", "rbf") or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Unknown kernel or invalid temperature")
    gram = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    if kernel == "linear":
        return gram
    return np.exp(tau * (np.clip(gram, -1., 1.) - 1.))


def fit_prepared(prepared, config, gram_matrices=None):
    """Weighted kernel ridge; mean-one balancing matches the earlier linear ridge."""
    penalty = config["l2"]
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Ridge penalty must be finite and positive")
    if prepared["transform"] != config["transform"]:
        raise ValueError("Prepared support transform differs from configuration")
    features = prepared["features"]
    n, classes, dimension = features.shape
    dual = np.zeros((n, classes), dtype=np.float64)
    directions = np.zeros((classes, dimension), dtype=np.float64)
    for c in np.flatnonzero(prepared["valid"]):
        gram = (kernel_matrix(features[:, c], features[:, c], config["kernel"], config["tau"])
                if gram_matrices is None else gram_matrices[c])
        root = np.sqrt(prepared["weights"][:, c])
        system = gram * np.outer(root, root)
        system.flat[::n + 1] += penalty
        dual[:, c] = root * np.linalg.solve(system, root * prepared["target"][:, c])
        if config["kernel"] == "linear":
            directions[c] = features[:, c].T @ dual[:, c]
    return {
        "config": dict(config), "mean": prepared["mean"], "features": features,
        "positive_counts": prepared["positive_counts"],
        "negative_counts": prepared["negative_counts"], "valid": prepared["valid"],
        "dual": dual, "directions": directions, "class_count": classes,
        "support_count": n,
    }


def fit_kernel(features, labels, baseline_logits, config):
    return fit_prepared(prepare_fit(features, labels, baseline_logits, config["transform"]), config)


def predict_kernel(state, query_features):
    """Only images/features enter this function; labels are not an argument."""
    query = transform_features(query_features, state["mean"])
    output = np.zeros((len(query), state["class_count"]), dtype=np.float64)
    config = state["config"]
    for c in np.flatnonzero(state["valid"]):
        if config["kernel"] == "linear":
            output[:, c] = query[:, c] @ state["directions"][c]
        else:
            gram = kernel_matrix(query[:, c], state["features"][:, c],
                                 config["kernel"], config["tau"])
            output[:, c] = gram @ state["dual"][:, c]
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite kernel residual")
    return output


def leave_group_out_grid(features, labels, baseline, groups, configurations,
                         deadline=None, progress=None):
    """Reuse fitting Gram matrices, never fitted means, across different folds."""
    labels = np.asarray(labels)
    features = np.asarray(features)
    baseline = np.asarray(baseline)
    groups = np.asarray(groups, dtype=str)
    if groups.shape != (len(labels),) or np.any(groups == "") or len(np.unique(groups)) < 2:
        raise ValueError("Need at least two nonempty support groups")
    keys = [config_key(config) for config in configurations]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError("Need nonempty unique configurations")
    outputs = {key: np.zeros_like(baseline, dtype=np.float64) for key in keys}
    audits = {}
    for transform in dict.fromkeys(config["transform"] for config in configurations):
        subset = [config for config in configurations if config["transform"] == transform]
        folds, missing_rows = [], np.zeros(labels.shape[1], dtype=np.int64)
        for group in np.unique(groups):
            check_deadline(deadline)
            held, kept = groups == group, groups != group
            prepared = prepare_fit(features[kept], labels[kept], baseline[kept], transform)
            x = prepared["features"]
            gram_by_kernel = {}
            for config in subset:
                check_deadline(deadline)
                kernel_key = (config["kernel"], config["tau"])
                if kernel_key not in gram_by_kernel:
                    gram_by_kernel[kernel_key] = [
                        kernel_matrix(x[:, c], x[:, c], *kernel_key)
                        if prepared["valid"][c] else None for c in range(labels.shape[1])]
                state = fit_prepared(prepared, config, gram_by_kernel[kernel_key])
                outputs[config_key(config)][held] = predict_kernel(state, features[held])
            missing_rows += (~prepared["valid"]).astype(np.int64) * int(held.sum())
            folds.append({
                "held_group_sha256": hashlib.sha256(group.encode()).hexdigest(),
                "held_indices": np.flatnonzero(held).tolist(),
                "fit_indices": np.flatnonzero(kept).tolist(),
                "fit_positive_counts": prepared["positive_counts"].tolist(),
                "fit_negative_counts": prepared["negative_counts"].tolist(),
                "zero_residual_classes": np.flatnonzero(~prepared["valid"]).tolist(),
                "fit_mean_sha256": array_digest(prepared["mean"]),
                "fit_balancing_weights_sha256": array_digest(prepared["weights"]),
                "fit_zero_feature_counts": prepared["zero_feature_counts"].tolist(),
            })
            if progress is not None:
                progress(transform, len(folds), len(np.unique(groups)))
        audits[transform] = {
            "fold_count": len(folds), "folds": folds,
            "missing_state_rows_per_class": missing_rows.tolist(),
            "mean_fitting": "Fold fitting support only; never held-out rows or test images",
        }
    return outputs, audits


def choose_candidates(records, top_k):
    """One OOF-selected configuration per transform/kernel, independent of test."""
    best = {}
    for record in records:
        config = record["config"]
        key = (config["transform"], config["kernel"])
        if key not in best or candidate_order(record) < candidate_order(best[key]):
            best[key] = record
    return sorted(best.values(), key=candidate_order)[:top_k]


def save_fit(path, state, support_identity):
    """Inference model retains RBF support feature centers, never raw labels."""
    np.savez_compressed(
        path, config_json=np.asarray(json.dumps(state["config"], sort_keys=True)),
        mean=state["mean"], support_features=state["features"], dual=state["dual"],
        directions=state["directions"], valid=state["valid"],
        positive_counts=state["positive_counts"], negative_counts=state["negative_counts"],
        support_identity=np.asarray(support_identity))


def load_fit(path):
    with np.load(path, allow_pickle=False) as data:
        state = {key: data[key].copy() for key in (
            "mean", "support_features", "dual", "directions", "valid",
            "positive_counts", "negative_counts")}
        state["config"] = json.loads(str(data["config_json"]))
    state["features"] = state.pop("support_features")
    state["support_count"], state["class_count"] = state["dual"].shape
    return state


def verified_inputs(cache, dataset=None):
    """Read support labels and label-free test metadata, checking cache provenance."""
    manifest_path, manifest = cache_view(cache)
    directory = manifest_path.parent
    name, split_dir = manifest["dataset"], Path(manifest["split_dir"])
    if dataset and name != dataset:
        raise ValueError("Dataset differs from frozen feature cache")
    from i2cl_data import load_spec
    support = load_spec(split_dir, name, "support")
    labels = np.asarray(support["labels"], dtype=np.int8)
    identities = {}
    for split in ("support", "test"):
        path = directory / f"{split}_metadata.json"
        if sha256(path) != manifest["auxiliary_files"][path.name]["sha256"]:
            raise ValueError("Cache metadata checksum mismatch")
        metadata = json.loads(path.read_text())
        if "labels" in metadata or fingerprint(metadata) != manifest[f"{split}_identity"]:
            raise ValueError("Cache split metadata identity mismatch")
        identities[split] = {
            **metadata, "csv_sha256": manifest["split_sha256"][
                str(split_dir / name / f"{split}.csv")]}
    if sha256(split_dir / name / "support.csv") != identities["support"]["csv_sha256"]:
        raise ValueError("Support CSV changed")
    for key in ("image_paths", "groups", "class_names", "prompts"):
        if support[key] != identities["support"][key]:
            raise ValueError(f"Support {key} differs from frozen cache")
    for key in ("image_paths", "groups", "ids"):
        if set(identities["support"][key]) & set(identities["test"][key]):
            raise ValueError(f"Support/test overlap in {key}")
    if list(support["class_names"]) != manifest["class_names"]:
        raise ValueError("Feature class columns changed")
    path = directory / "support_labels.npy"
    if (sha256(path) != manifest["auxiliary_files"][path.name]["sha256"]
            or not np.array_equal(labels, np.load(path, allow_pickle=False))):
        raise ValueError("Support labels differ from cache")
    return manifest_path, manifest, support, labels, identities


def load_array(manifest_path, manifest, split, key):
    record = manifest["splits"][split]["files"][key]
    path = manifest_path.parent / record["file"]
    if sha256(path) != record["sha256"]:
        raise ValueError("Frozen feature checksum mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
        raise ValueError("Frozen feature shape/dtype mismatch")
    return values


def run(args):
    with threadpool_limits(limits=args.threads):
        return _run(args)


def _run(args):
    manifest_path, manifest, support, labels, identities = verified_inputs(args.cache, args.dataset)
    dataset, split_dir = manifest["dataset"], Path(manifest["split_dir"])
    baseline = np.asarray(load_array(manifest_path, manifest, "support", "base_logits"),
                          dtype=np.float64)
    feature = load_array(manifest_path, manifest, "support", "class_local")
    if baseline.shape != labels.shape or feature.shape[:2] != labels.shape:
        raise ValueError("Support feature/baseline axes differ from labels")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    configurations = build_grid(args)
    source_paths = [Path(__file__).resolve()] + [
        Path(__file__).with_name(name) for name in (
            "overnight_cache_methods.py", "overnight_oof_loss_tiebreak.py",
            "batch_metrics.py", "i2cl_data.py")]
    write_json(output / "run_config.json", {
        "version": VERSION, "dataset": dataset, "created_at": started, "arguments": vars(args),
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": sha256(manifest_path),
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "support_csv_sha256": identities["support"]["csv_sha256"],
        "test_csv_sha256": identities["test"]["csv_sha256"],
        "support_identity": manifest["support_identity"], "test_identity": manifest["test_identity"],
        "checkpoint": manifest.get("checkpoint"),
        "root_split_fingerprint": manifest.get("root_split_fingerprint"),
        "support_count": len(labels), "class_names": support["class_names"],
        "fit_grid": configurations, "feature": "class_local",
        "fit_objective": "Class-balanced weighted kernel ridge of y - sigmoid(frozen base logits)",
        "weighting": "Fold-fit negatives/positives for positive rows; weights normalized to mean 1",
        "selection_rule": SELECTION_RULE, "test_labels_used_in_fitting_or_selection": False,
        "interpretation": "Exploratory score residual; not I2CL; unchanged maps do not explain residual",
    })
    write_json(output / "status.json", {"status": "support_selection", "started_at": started})
    try:
        def progress(transform, count, total):
            write_json(output / "status.json", {
                "status": "support_selection", "transform": transform, "folds_done": count,
                "folds_total": total, "updated_at": datetime.now(timezone.utc).isoformat()})
            if count == 1 or count % 10 == 0 or count == total:
                print(f"{dataset}: {transform} support OOF {count}/{total}", flush=True)

        residuals, audits = leave_group_out_grid(
            feature, labels, baseline, support["groups"], configurations, args.deadline, progress)
        spec = {"name": dataset, "labels": labels, "class_names": support["class_names"]}
        baseline_metrics = compute_metrics(spec, baseline)
        folds = {}
        for transform, audit in audits.items():
            folds[transform] = build_bce_folds(labels, support["groups"], audit)
            audit["bce_folds"] = folds[transform]
        first_folds = next(iter(folds.values()))
        baseline_bce = weighted_oof_bce(baseline, labels, first_folds)
        records = []
        for config in configurations:
            check_deadline(args.deadline)
            key = config_key(config)
            for strength in args.strengths:
                logits = baseline + strength * residuals[key]
                metrics = compute_metrics(spec, logits)
                metrics["score_type"] = "OOF RadZero logits plus weighted kernel residual"
                bce = weighted_oof_bce(logits, labels, folds[config["transform"]])
                records.append({
                    "candidate_id": f"{key}_strength{strength:g}",
                    "family": family_name(config),
                    "config": {**config, "method": "weighted_kernel_ridge",
                               "feature": "class_local", "strength": strength},
                    "support_oof_metrics": metrics, "support_oof_bce": bce,
                    "support_oof_delta": metric_delta(baseline_metrics, metrics),
                    "selection_score": metric_score(metrics), "fold_audit_key": config["transform"],
                })
        np.savez_compressed(output / "oof_residuals.npz", **residuals)
        write_json(output / "fold_audits.json", audits)
        write_json(output / "candidates.json", records)
        chosen = choose_candidates(records, args.top_k)
        selection = {
            "selected_before_test_labels_loaded": True, "selection_time": time.time(),
            "selection_rule": SELECTION_RULE, "primary_tie_policy": "Exact floating-point equality",
            "baseline_support_metrics": baseline_metrics, "baseline_support_oof_bce": baseline_bce,
            "support_labels_sha256": array_digest(labels),
            "support_groups_sha256": hashlib.sha256(json.dumps(support["groups"]).encode()).hexdigest(),
            "oof_residuals_sha256": sha256(output / "oof_residuals.npz"),
            "fold_audits_sha256": sha256(output / "fold_audits.json"),
            "candidate_count": len(records), "top_k_distinct_transform_kernel": args.top_k,
            "candidates": chosen, "zero_strength_retained": 0. in args.strengths,
            "support_selected_candidate_id": chosen[0]["candidate_id"],
        }
        write_json(output / "selection.json", selection)
        selection_hash = sha256(output / "selection.json")
        write_json(output / "status.json", {
            "status": "selected_test_prediction", "selection_sha256": selection_hash})
        # Test feature arrays and labels are first consumed after immutable selection.
        baseline_test = np.asarray(
            load_array(manifest_path, manifest, "test", "base_logits"), dtype=np.float64)
        test_features = load_array(manifest_path, manifest, "test", "class_local")
        expected_shape = (len(identities["test"]["image_paths"]), labels.shape[1])
        if baseline_test.shape != expected_shape or test_features.shape[:2] != expected_shape:
            raise ValueError("Test features/baseline differ from metadata axes")
        results = []
        for record in chosen:
            check_deadline(args.deadline)
            config = record["config"]
            destination = output / record["candidate_id"]
            destination.mkdir()
            state = fit_kernel(feature, labels, baseline, config)
            save_fit(destination / "fit.npz", state, identities["support"]["csv_sha256"])
            reloaded = load_fit(destination / "fit.npz")
            predictions = np.empty_like(baseline_test)
            for first in range(0, len(predictions), args.chunk_size):
                check_deadline(args.deadline)
                last = min(len(predictions), first + args.chunk_size)
                residual = predict_kernel(reloaded, test_features[first:last])
                predictions[first:last] = baseline_test[first:last] + config["strength"] * residual
            np.save(destination / "predictions.npy", predictions, allow_pickle=False)
            np.save(destination / "oof_logits.npy",
                    baseline + config["strength"] * residuals[config_key(config)], allow_pickle=False)
            if sha256(output / "selection.json") != selection_hash:
                raise RuntimeError("Selection changed after reading test features")
            check_deadline(args.deadline)
            evaluation = evaluate_test_only(
                split_dir, dataset, predictions, baseline_test, identities["test"], destination)
            result = {
                **record, **evaluation, "selection_sha256": selection_hash,
                "support_selected": record["candidate_id"] == chosen[0]["candidate_id"],
                "fit_sha256": sha256(destination / "fit.npz"),
                "prediction_sha256": sha256(destination / "predictions.npy"),
                "oof_logits_sha256": sha256(destination / "oof_logits.npy"),
            }
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
        write_json(output / "status.json", {
            "status": "complete", "selected_count": len(results),
            "completed_at": datetime.now(timezone.utc).isoformat()})
    except DeadlineReached as error:
        write_json(output / "status.json", {"status": "deadline", "reason": str(error)})
        raise
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--transforms", nargs="+", choices=TRANSFORMS, default=list(TRANSFORMS))
    parser.add_argument("--kernels", nargs="+", choices=("linear", "rbf"), default=["linear", "rbf"])
    parser.add_argument("--linear-l2s", type=float, nargs="+", default=[.01, .1, 1., 10.])
    parser.add_argument("--rbf-l2s", type=float, nargs="+", default=[.1, 1.])
    parser.add_argument("--taus", type=float, nargs="+", default=[1., 5., 20., 100.])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0., .3, 1., 3., 10.])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=parse_deadline)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4 or args.top_k < 1 or args.chunk_size < 1:
        parser.error("threads must be 1..4; top-k and chunk-size must be positive")
    if (any(not math.isfinite(x) or x < 0 for x in args.strengths)
            or 0. not in args.strengths or len(set(args.strengths)) != len(args.strengths)):
        parser.error("strengths must be unique, finite, nonnegative and include zero")
    if any(not math.isfinite(x) or x <= 0 for x in args.linear_l2s + args.rbf_l2s + args.taus):
        parser.error("Penalties and temperatures must be finite and positive")
    return args


if __name__ == "__main__":
    run(parse_args())
