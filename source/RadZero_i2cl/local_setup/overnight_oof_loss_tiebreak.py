#!/usr/bin/env python3
"""Support-only cache methods with weighted OOF BCE to break exact rank ties.

Primary selection is mean(macro AUROC, macro AUPRC); only exact ties consult
lower fold-weighted BCE, then lower residual strength. This independent
ablation never changes the stable cache worker or fits using test labels.
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
    FEATURES, DeadlineReached, array_digest, cache_view, check_deadline,
    evaluate_test_only, fingerprint, fit_residual, leave_group_out, metric_delta,
    metric_score, normalized_features, parse_deadline, predict_residual,
    save_fit, sha256, write_json,
)

VERSION = "radzero-frozen-cache-oof-weighted-bce-tiebreak-v1"
SELECTION_RULE = (
    "Descending mean(macro_AUROC, macro_AUPRC); exact ties only: ascending "
    "held-out support BCE with fold-fit negatives/positives; then ascending strength"
)


def build_bce_folds(labels, groups, fold_audit):
    """Derive BCE weights solely from each fold's fitting rows, never its holdout."""
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=str)
    if labels.ndim != 2 or groups.shape != (len(labels),):
        raise ValueError("Expected [N,C] labels and N support groups")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Support labels must be binary")
    seen = np.zeros(len(labels), dtype=int)
    folds = []
    universe = set(range(len(labels)))
    for index, original in enumerate(fold_audit["folds"]):
        fit = np.asarray(original["fit_indices"], dtype=int)
        held = np.asarray(original["held_indices"], dtype=int)
        if not len(fit) or not len(held):
            raise ValueError("Empty fitting/held-out fold")
        if (len(set(fit)) != len(fit) or len(set(held)) != len(held)
                or set(fit) & set(held) or set(fit) | set(held) != universe):
            raise ValueError("Fold rows must be a disjoint complete partition")
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Related support groups cross fit/held-out folds")
        seen[held] += 1
        positive = labels[fit].sum(axis=0).astype(int)
        negative = len(fit) - positive
        valid = (positive > 0) & (negative > 0)
        if (positive.tolist() != original["fit_positive_counts"]
                or negative.tolist() != original["fit_negative_counts"]
                or np.flatnonzero(~valid).tolist() != original["zero_residual_classes"]):
            raise ValueError("OOF fit audit differs from actual fitting labels")
        weights = np.divide(negative, positive, out=np.zeros(labels.shape[1]),
                            where=valid)
        folds.append({
            "fold_index": index, "fit_indices": fit.tolist(),
            "held_indices": held.tolist(),
            "held_group_sha256": original["held_group_sha256"],
            "fit_positive_counts": positive.tolist(),
            "fit_negative_counts": negative.tolist(),
            "valid_classes": valid.tolist(),
            "pos_weight": [float(weights[c]) if valid[c] else None
                           for c in range(labels.shape[1])],
            "effective_entries": int(len(held) * valid.sum()),
            "excluded_entries": int(len(held) * (~valid).sum()),
        })
    if not np.all(seen == 1):
        raise ValueError("Every support row must be held out exactly once")
    return folds


def weighted_oof_bce(logits, labels, folds):
    """Stable BCE directly from raw logits, excluding fold-missing classes."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits matching labels")
    total, effective, excluded = 0., 0, 0
    per_class = np.zeros(labels.shape[1], dtype=int)
    details = []
    for fold in folds:
        held = np.asarray(fold["held_indices"], dtype=int)
        valid = np.asarray(fold["valid_classes"], dtype=bool)
        count = int(len(held) * valid.sum())
        skipped = int(len(held) * (~valid).sum())
        weights = np.asarray([value for value in fold["pos_weight"]
                              if value is not None], dtype=np.float64)
        z, y = logits[held][:, valid], labels[held][:, valid]
        # torch BCEWithLogitsLoss(pos_weight=weights), reduction=sum,
        # expressed explicitly without sigmoid or clipping raw scores.
        losses = y * weights[None] * np.logaddexp(0., -z) + (1 - y) * np.logaddexp(0., z)
        subtotal = float(losses.sum())
        total += subtotal
        effective += count
        excluded += skipped
        per_class += valid.astype(int) * len(held)
        details.append({
            "fold_index": fold["fold_index"], "weighted_bce_sum": subtotal,
            "weighted_bce": subtotal / count if count else None,
            "effective_entries": count, "excluded_entries": skipped,
        })
    return {
        "weighted_bce": total / effective if effective else None,
        "weighted_bce_sum": total, "effective_entries": effective,
        "excluded_entries": excluded,
        "effective_entries_per_class": per_class.tolist(), "folds": details,
        "reduction": "sum of valid held-out weighted BCE / valid held-out entries",
        "weight_source": "negatives/positives from each fold's fitting support only",
        "input": "raw RadZero logits plus raw score residual; no sigmoid preprocessing",
    }


def candidate_order(record):
    bce = record["support_oof_bce"]["weighted_bce"]
    return (-record["selection_score"], math.inf if bce is None else bce,
            record["config"]["strength"], record["candidate_id"])


def choose_candidates(records, top_k):
    """One winner per method/feature; exact primary ties use OOF BCE."""
    best = {}
    for record in records:
        key = (record["config"]["method"], record["config"]["feature"])
        if key not in best or candidate_order(record) < candidate_order(best[key]):
            best[key] = record
    return sorted(best.values(), key=candidate_order)[:top_k]


def run(args):
    # Explicit independent run path; immutable shared numerical helpers only.
    manifest_path, manifest = cache_view(args.cache)
    cache_dir = manifest_path.parent
    dataset, split_dir = manifest["dataset"], Path(manifest["split_dir"])
    if args.dataset and args.dataset != dataset:
        raise ValueError("Dataset differs from feature cache")
    from i2cl_data import load_spec
    support = load_spec(split_dir, dataset, "support")
    labels = np.asarray(support["labels"], dtype=np.int8)
    classes = len(support["class_names"])

    def load_array(split, name):
        keys = {
            "baseline_logits": ("base_logits", None), "class_local": ("class_local", None),
            "dino_cls": ("global_dino", 0), "aligned_cls": ("global_aligned", 0),
            "dino_mean": ("global_dino", 1), "aligned_mean": ("global_aligned", 1),
        }
        key, index = keys[name]
        record = manifest["splits"][split]["files"][key]
        path = cache_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Frozen feature checksum mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError("Frozen feature shape/dtype mismatch")
        if index is not None:
            if array.ndim != 3 or array.shape[1] != 2:
                raise ValueError("Global features must be [N,2,D]")
            return array[:, index]
        return array

    identities = {}
    for split in ("support", "test"):
        path = cache_dir / f"{split}_metadata.json"
        if sha256(path) != manifest["auxiliary_files"][path.name]["sha256"]:
            raise ValueError("Cache metadata checksum mismatch")
        metadata = json.loads(path.read_text())
        if "labels" in metadata or fingerprint(metadata) != manifest[f"{split}_identity"]:
            raise ValueError("Cache split metadata identity mismatch")
        identities[split] = {
            **metadata, "csv_sha256": manifest["split_sha256"][
                str(split_dir / dataset / f"{split}.csv")]}
    support_identity, test_identity = identities["support"], identities["test"]
    if sha256(split_dir / dataset / "support.csv") != support_identity["csv_sha256"]:
        raise ValueError("Support CSV changed")
    for key in ("image_paths", "groups", "class_names", "prompts"):
        if support[key] != support_identity[key]:
            raise ValueError(f"Support {key} differs from cache")
    for key in ("image_paths", "groups", "ids"):
        if set(support_identity[key]) & set(test_identity[key]):
            raise ValueError(f"Support/test overlap in {key}")
    if list(support["class_names"]) != manifest["class_names"]:
        raise ValueError("Class columns changed")
    label_path = cache_dir / "support_labels.npy"
    if (sha256(label_path) != manifest["auxiliary_files"][label_path.name]["sha256"]
            or not np.array_equal(labels, np.load(label_path, allow_pickle=False))):
        raise ValueError("Support label cache mismatch")
    baseline = np.asarray(load_array("support", "baseline_logits"), dtype=np.float64)
    if baseline.shape != labels.shape:
        raise ValueError("Support baseline shape mismatch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    source_paths = [Path(__file__).resolve()] + [
        Path(__file__).with_name(name) for name in ("overnight_cache_methods.py", "batch_metrics.py",
                                                   "i2cl_data.py")]
    write_json(output / "run_config.json", {
        "version": VERSION, "dataset": dataset, "created_at": started,
        "arguments": vars(args), "selection_rule": SELECTION_RULE,
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": sha256(manifest_path),
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "support_csv_sha256": support_identity["csv_sha256"],
        "test_csv_sha256": test_identity["csv_sha256"], "support_count": len(labels),
        "class_names": support["class_names"], "test_labels_used_in_fitting_or_selection": False,
        "interpretation": "Exploratory OOF selection ablation; score residual, not original I2CL; no adapted maps",
    })
    write_json(output / "status.json", {"status": "support_selection", "started_at": started})
    spec = {"name": dataset, "labels": labels, "class_names": support["class_names"]}
    baseline_metrics = compute_metrics(spec, baseline)
    features, records, audits = {}, [], {}
    baseline_bce = None
    try:
        for feature_name in args.features:
            check_deadline(args.deadline)
            feature = normalized_features(load_array("support", feature_name), classes)
            features[feature_name] = feature
            for method in args.methods:
                parameters = args.taus if method == "cache" else (
                    args.l2s if method == "ridge" else [1.])
                for parameter in parameters:
                    check_deadline(args.deadline)
                    residual, audit = leave_group_out(
                        feature, labels, baseline, support["groups"], method, parameter, args.deadline)
                    bce_folds = build_bce_folds(labels, support["groups"], audit)
                    if baseline_bce is None:
                        baseline_bce = weighted_oof_bce(baseline, labels, bce_folds)
                    key = f"{feature_name}_{method}_{parameter:g}"
                    audits[key] = {**audit, "bce_folds": bce_folds}
                    for strength in args.strengths:
                        logits = baseline + strength * residual
                        metrics = compute_metrics(spec, logits)
                        metrics["score_type"] = "OOF raw RadZero logits plus support-only score residual"
                        bce = weighted_oof_bce(logits, labels, bce_folds)
                        records.append({
                            "candidate_id": f"{key}_strength{strength:g}",
                            "config": {"method": method, "feature": feature_name,
                                       "parameter": parameter, "strength": strength},
                            "support_oof_metrics": metrics,
                            "support_oof_delta": metric_delta(baseline_metrics, metrics),
                            "support_oof_bce": bce, "selection_score": metric_score(metrics),
                            "fold_audit_key": key,
                        })
                    write_json(output / "candidates.json", records)
                    print(f"{dataset}: BCE tie-break OOF {key}; {len(records)} candidates", flush=True)
        write_json(output / "fold_audits.json", audits)
        chosen = choose_candidates(records, args.top_k)
        selection = {
            "selected_before_test_labels_loaded": True, "selection_time": time.time(),
            "selection_rule": SELECTION_RULE, "primary_tie_policy": "Exact floating-point equality",
            "baseline_support_metrics": baseline_metrics, "baseline_support_oof_bce": baseline_bce,
            "support_labels_sha256": array_digest(labels),
            "support_groups_sha256": hashlib.sha256(json.dumps(support["groups"]).encode()).hexdigest(),
            "top_k_distinct_method_feature": args.top_k, "candidates": chosen,
            "zero_strength_retained": 0. in args.strengths,
        }
        write_json(output / "selection.json", selection)
        selection_hash = sha256(output / "selection.json")
        write_json(output / "status.json", {"status": "selected_test_prediction",
                                           "selection_sha256": selection_hash})
        baseline_test = np.asarray(load_array("test", "baseline_logits"), dtype=np.float64)
        results = []
        for record in chosen:
            check_deadline(args.deadline)
            config = record["config"]
            destination = output / record["candidate_id"]
            destination.mkdir()
            state = fit_residual(features[config["feature"]], labels, baseline,
                                 config["method"], config["parameter"])
            save_fit(destination / "fit.npz", state, support_identity["csv_sha256"])
            query = load_array("test", config["feature"])
            logits = np.empty_like(baseline_test)
            for first in range(0, len(logits), args.chunk_size):
                check_deadline(args.deadline)
                last = min(len(logits), first + args.chunk_size)
                x = normalized_features(query[first:last], classes)
                logits[first:last] = baseline_test[first:last] + config["strength"] * predict_residual(state, x)
            np.save(destination / "predictions.npy", logits, allow_pickle=False)
            if sha256(output / "selection.json") != selection_hash:
                raise RuntimeError("Selection changed after test feature reading")
            check_deadline(args.deadline)
            evaluation = evaluate_test_only(split_dir, dataset, logits, baseline_test,
                                            test_identity, destination)
            result = {
                **record, **evaluation, "selection_sha256": selection_hash,
                "prediction_sha256": sha256(destination / "predictions.npy"),
                "fit_sha256": sha256(destination / "fit.npz"),
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
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "selected_count": len(results)})
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
    parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                        choices=("cache", "proto", "ridge"), default=["cache", "proto", "ridge"])
    parser.add_argument("--features", nargs="+", choices=FEATURES,
                        default=["class_local", "dino_cls", "aligned_cls"])
    parser.add_argument("--taus", nargs="+", type=float, default=[1., 5., 10., 20.])
    parser.add_argument("--l2s", nargs="+", type=float, default=[.1, 1., 10.])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0., .1, .3, 1., 3.])
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=parse_deadline)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4 or args.top_k < 1 or args.chunk_size < 1:
        parser.error("threads must be 1..4; top-k/chunk-size must be positive")
    if any(not math.isfinite(x) or x < 0 for x in args.strengths) or 0. not in args.strengths:
        parser.error("strengths must be finite/nonnegative and retain strength 0")
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
