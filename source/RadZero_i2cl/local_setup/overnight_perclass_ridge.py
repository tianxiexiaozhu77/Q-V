#!/usr/bin/env python3
"""Class-local ridge, selecting each class solely by grouped support OOF scores."""

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics
from overnight_cache_methods import (
    DeadlineReached, array_digest, check_deadline, evaluate_test_only, fit_residual,
    leave_group_out, normalized_features, parse_deadline, sha256, write_json,
)
from overnight_linear_residual import FrozenCache
from overnight_oof_loss_tiebreak import build_bce_folds, weighted_oof_bce

FAMILY = "class_local_perclass_ridge"
VERSION = "support-group-oof-perclass-weighted-ridge-v1"
PENALTIES = [.01, .1, 1., 10., 100.]
STRENGTHS = [0., .3, 1., 3., 10., 30.]
RULE = (
    "Per-class descending mean(AUROC,AUPRC); exact ties ascending fold-weighted BCE; "
    "then smaller strength, then stronger ridge regularization"
)


def class_bce_folds(folds, index):
    return [{
        **fold, "valid_classes": [fold["valid_classes"][index]],
        "pos_weight": [fold["pos_weight"][index]],
    } for fold in folds]


def class_candidate_order(record):
    loss = record["oof_bce"]["weighted_bce"]
    return (-record["selection_score"], math.inf if loss is None else loss,
            record["strength"], -record["parameter"], record["candidate_id"])


def select_per_class(records, class_names):
    chosen = []
    for index, name in enumerate(class_names):
        choices = [record["per_class"][index] for record in records]
        # An unidentifiable class receives no adaptation. Do not extrapolate
        # labels from another class or from a held-out test image.
        fallback = not any(choice["oof_bce"]["effective_entries"] for choice in choices)
        if fallback:
            choices = [choice for choice in choices if choice["strength"] == 0.]
        best = min(choices, key=class_candidate_order)
        chosen.append({**best, "class_index": index, "class_name": name,
                       "zero_state_fallback": fallback})
    return chosen


def support_search(features, labels, baseline, groups, class_names, penalties, strengths,
                   deadline=None, progress=None):
    records, audits, oof_arrays = [], {}, {}
    spec = {"name": "support", "labels": labels, "class_names": class_names}
    for penalty in penalties:
        check_deadline(deadline)
        residual, audit = leave_group_out(
            features, labels, baseline, groups, "ridge", penalty, deadline)
        folds = build_bce_folds(labels, groups, audit)
        key = f"class_local_ridge_{penalty:g}"
        audits[key] = {**audit, "bce_folds": folds}
        for strength in strengths:
            candidate_id = f"{key}_strength{strength:g}"
            logits = baseline + strength * residual
            metrics = compute_metrics(spec, logits)
            per_class = []
            for index, name in enumerate(class_names):
                metric = metrics["per_class"][name]
                values = [value for value in (metric["auroc"], metric["auprc"])
                          if value is not None]
                bce = weighted_oof_bce(
                    logits[:, index:index + 1], labels[:, index:index + 1],
                    class_bce_folds(folds, index))
                per_class.append({
                    "class_name": name, "class_index": index, "candidate_id": candidate_id,
                    "parameter": float(penalty), "strength": float(strength),
                    "auroc": metric["auroc"], "auprc": metric["auprc"],
                    "selection_score": sum(values) / len(values) if values else 0.,
                    "oof_bce": bce, "fold_audit_key": key,
                })
            records.append({"candidate_id": candidate_id, "parameter": float(penalty),
                            "strength": float(strength), "per_class": per_class,
                            "support_oof_metrics": metrics})
            oof_arrays[candidate_id] = logits
        if progress:
            progress(records, audits)
    choices = select_per_class(records, class_names)
    assembled = np.empty_like(baseline, dtype=np.float64)
    for index, choice in enumerate(choices):
        assembled[:, index] = oof_arrays[choice["candidate_id"]][:, index]
    return records, audits, choices, assembled, oof_arrays


def fit_selected(features, labels, baseline, choices):
    directions = np.zeros((labels.shape[1], features.shape[-1]), dtype=np.float64)
    strengths = np.zeros(labels.shape[1], dtype=np.float64)
    states = {}
    for index, choice in enumerate(choices):
        penalty = choice["parameter"]
        if penalty not in states:
            states[penalty] = fit_residual(features, labels, baseline, "ridge", penalty)
        state = states[penalty]
        if state["valid"][index] and not choice["zero_state_fallback"]:
            directions[index] = state["directions"][index]
            strengths[index] = choice["strength"]
    return directions, strengths


def predict_selected(features, baseline, directions, strengths):
    baseline = np.asarray(baseline, dtype=np.float64)
    if features.shape != (len(baseline), baseline.shape[1], directions.shape[1]):
        raise ValueError("Expected normalized class-local [N,C,D] features")
    residual = np.empty_like(baseline)
    for index in range(baseline.shape[1]):
        residual[:, index] = features[:, index] @ directions[index]
    result = baseline + strengths[None] * residual
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite assembled predictions")
    return result


def check_binary_selection(reference_root, dataset, choice):
    directory = Path(reference_root) / dataset
    selection = json.loads((directory / "selection.json").read_text())
    candidate = next(row for row in selection["candidates"]
                     if row["config"]["method"] == "ridge"
                     and row["config"]["feature"] == "class_local")
    if (candidate["config"]["parameter"] != choice["parameter"]
            or candidate["config"]["strength"] != choice["strength"]):
        raise ValueError("Binary per-class OOF selection differs from global class_local rule")
    return {
        "passed": True, "reference": str(directory),
        "candidate_id": candidate["candidate_id"],
        "selection_sha256": sha256(directory / "selection.json"),
        "parameter": choice["parameter"], "strength": choice["strength"],
    }


def run(args):
    cache = FrozenCache(args.cache)
    if args.dataset and cache.dataset != args.dataset:
        raise ValueError("Dataset/cache mismatch")
    support = cache.support()
    labels = np.asarray(support["labels"], dtype=np.int8)
    baseline = np.asarray(cache.array("support", "baseline_logits"), dtype=np.float64)
    features = normalized_features(cache.array("support", "class_local"), labels.shape[1])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "support_selection", "started_at": time.time()})
    write_json(output / "run_config.json", {
        "version": VERSION, "family": FAMILY, "dataset": cache.dataset,
        "arguments": vars(args), "selection_rule": RULE,
        "cache_manifest": str(cache.path), "cache_manifest_sha256": sha256(cache.path),
        "support_csv_sha256": cache.metadata["support"]["csv_sha256"],
        "test_csv_sha256": cache.metadata["test"]["csv_sha256"],
        "support_count": len(labels), "source_sha256": sha256(__file__),
        "class_names": support["class_names"],
        "test_labels_used_for_fitting_or_selection": False,
        "interpretation": "Exploratory support-selected per-class rule; not test-selected column splicing; original maps do not explain residual",
    })
    try:
        def progress(records, audits):
            write_json(output / "candidates.json", records)
            write_json(output / "fold_audits.json", audits)
            print(f"{cache.dataset}: {len(records)} per-class OOF candidates", flush=True)
        records, audits, choices, assembled, oof_arrays = support_search(
            features, labels, baseline, support["groups"], support["class_names"],
            args.l2s, args.strengths, args.deadline, progress)
        binary_check = None
        if labels.shape[1] == 1 and args.binary_reference_root:
            binary_check = check_binary_selection(
                args.binary_reference_root, cache.dataset, choices[0])
        selection = {
            "family": FAMILY, "selection_rule": RULE, "per_class": choices,
            "selected_before_test_labels_loaded": True, "selection_time": time.time(),
            "support_labels_sha256": array_digest(labels),
            "support_groups_sha256": array_digest(np.asarray(support["groups"])),
            "assembled_support_oof_metrics": compute_metrics(support, assembled),
            "binary_global_selection_check": binary_check,
            "warning": "Per-class choices are entirely support-derived; aggregate OOF is selected, not an unbiased validation estimate",
        }
        write_json(output / "selection.json", selection)
        selection_sha = sha256(output / "selection.json")
        np.savez_compressed(output / "support_oof_logits.npz", **oof_arrays)
        directions, strengths = fit_selected(features, labels, baseline, choices)
        np.savez(output / "perclass_ridge.npz", directions=directions, strengths=strengths,
                 penalties=np.asarray([row["parameter"] for row in choices]),
                 class_names=np.asarray(support["class_names"]), prompts=np.asarray(support["prompts"]),
                 support_sha256=np.asarray(cache.metadata["support"]["csv_sha256"]))
        check_deadline(args.deadline)
        base_test = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float64)
        raw = cache.array("test", "class_local")
        predictions = np.empty_like(base_test)
        for start in range(0, len(base_test), args.chunk_size):
            check_deadline(args.deadline)
            stop = min(len(base_test), start + args.chunk_size)
            normalized = normalized_features(raw[start:stop], labels.shape[1])
            predictions[start:stop] = predict_selected(
                normalized, base_test[start:stop], directions, strengths)
        np.save(output / "predictions.npy", predictions, allow_pickle=False)
        if binary_check:
            reference = np.load(
                Path(binary_check["reference"]) / binary_check["candidate_id"] / "predictions.npy",
                allow_pickle=False)
            if not np.array_equal(reference, predictions):
                error = float(np.abs(reference - predictions).max())
                raise ValueError(f"Binary per-class predictions differ from global reference: {error}")
            binary_check = {**binary_check, "predictions_bitwise_equal": True,
                            "test_prediction_labels_read": False}
            write_json(output / "binary_equivalence.json", binary_check)
        if sha256(output / "selection.json") != selection_sha:
            raise RuntimeError("Selection changed after test feature prediction")
        check_deadline(args.deadline)
        evaluation = evaluate_test_only(
            cache.split_dir, cache.dataset, predictions, base_test, cache.metadata["test"], output)
        result = {
            **evaluation, "family": FAMILY, "candidate_id": FAMILY,
            "config": {"method": "ridge_perclass", "feature": "class_local",
                       "parameters": [row["parameter"] for row in choices],
                       "strengths": [row["strength"] for row in choices]},
            "selection_sha256": selection_sha, "binary_equivalence": binary_check,
            "prediction_sha256": sha256(output / "predictions.npy"),
            "model_sha256": sha256(output / "perclass_ridge.npz"),
        }
        write_json(output / "result.json", result)
        write_json(output / "summary.json", {
            "dataset": cache.dataset, "family": FAMILY, "exploratory": True,
            "selection": selection, "results": [result],
            "test_labels_used_for_fitting_or_selection": False,
        })
        write_json(output / "status.json", {
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
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
    parser.add_argument("--l2s", type=float, nargs="+", default=PENALTIES)
    parser.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--deadline", type=parse_deadline)
    parser.add_argument("--binary-reference-root")
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4 or args.chunk_size < 1:
        parser.error("Invalid threads/chunk-size")
    if (any(not math.isfinite(x) or x <= 0 for x in args.l2s)
            or any(not math.isfinite(x) or x < 0 for x in args.strengths)
            or 0. not in args.strengths):
        parser.error("Need positive penalties, nonnegative strengths, and strength zero")
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
