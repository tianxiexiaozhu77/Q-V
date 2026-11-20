#!/usr/bin/env python3
"""Closed-form, support-only pairwise margin residual on class-local features.

Each class minimizes mean_{p,n}[(b_p-b_n)+(x_p-x_n)w-m]^2 + lambda*||w||².
Every support score used for selection is predicted with its entire group
excluded. Selection uses full OOF ranking metrics; exact ties use a common
squared-hinge pair risk (reference margin=1), then smaller residual strength.
OOF pairs can contain predictions from different fitted folds, so fold-specific
calibration differences can influence the selection risk.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics
from overnight_cache_methods import (
    array_digest, evaluate_test_only, metric_delta, metric_score,
    normalized_features, sha256, write_json,
)
from overnight_linear_residual import FrozenCache

VERSION = "radzero-closed-form-pairwise-margin-residual-v1"
MARGINS = (.25, 1., 4.)
PENALTIES = (.001, .01, .1, 1.)
STRENGTHS = (0., .25, .5, 1., 2., 4.)
HARD_DEADLINE_TEXT = "2026-10-21T10:00:00+08:00"
HARD_DEADLINE = datetime.fromisoformat(HARD_DEADLINE_TEXT).timestamp()
RISK_MARGIN = 1.


class StopRequested(RuntimeError):
    pass


def check_limits(deadline=None, stop_file=None):
    if deadline is not None and time.time() >= deadline:
        raise StopRequested("Authorized deadline reached")
    if stop_file is not None and Path(stop_file).exists():
        raise StopRequested(f"Campaign stop file exists: {stop_file}")


def pair_system(labels):
    """L=mean(D.T D), q=mean(D.T 1) for all positive-minus-negative pairs."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Pair labels must be a binary vector")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        return None
    p, n = len(positive), len(negative)
    laplacian = np.zeros((len(labels), len(labels)), dtype=np.float64)
    laplacian[positive, positive] = 1. / p
    laplacian[negative, negative] = 1. / n
    laplacian[np.ix_(positive, negative)] = -1. / (p * n)
    laplacian[np.ix_(negative, positive)] = -1. / (p * n)
    q = np.where(labels == 1, 1. / p, -1. / n)
    return laplacian, q


def class_solutions(features, labels, baseline, penalties, gram=None):
    """Two directions per lambda: w(m)=m*w_margin+w_base, no test inputs."""
    x, b = np.asarray(features, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2 or b.shape != labels.shape or len(x) != len(labels):
        raise ValueError("Class support arrays differ")
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite support inputs")
    if any(not np.isfinite(value) or value <= 0 for value in penalties):
        raise ValueError("Ridge penalties must be finite and positive")
    structure = pair_system(labels)
    if structure is None:
        return {penalty: np.zeros((x.shape[1], 2)) for penalty in penalties}
    laplacian, q = structure
    kernel = x @ x.T if gram is None else np.asarray(gram, dtype=np.float64)
    if kernel.shape != (len(x), len(x)) or not np.isfinite(kernel).all():
        raise ValueError("Invalid support Gram matrix")
    rhs = np.column_stack((q, -laplacian @ b))
    result = {}
    for penalty in penalties:
        # w=X.T a and (L K + lambda I)a = m*q-L*b gives the unique
        # primal optimum even when K is rank deficient; lambda is positive.
        system = laplacian @ kernel + penalty * np.eye(len(x))
        dual = np.linalg.solve(system, rhs)
        np.testing.assert_allclose(system @ dual, rhs, atol=1e-9, rtol=1e-9)
        directions = x.T @ dual
        if not np.isfinite(directions).all():
            raise ValueError("Nonfinite pairwise fit")
        result[penalty] = directions
    return result


def validate_support(features, labels, baseline):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    b = np.asarray(baseline, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 2 or x.shape[:2] != y.shape or b.shape != y.shape:
        raise ValueError("Need features [N,C,D] and labels/baseline [N,C]")
    if not len(y) or not np.isin(y, (0, 1)).all():
        raise ValueError("Support labels must be nonempty and binary")
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite support features/baseline")
    return x, y.astype(np.int8), b


def fit_pairwise(features, labels, baseline, margin, penalty):
    x, y, b = validate_support(features, labels, baseline)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("Pair margin must be positive")
    positives = y.sum(axis=0).astype(int)
    negatives = len(y) - positives
    valid = (positives > 0) & (negatives > 0)
    weights = np.zeros((y.shape[1], x.shape[2]), dtype=np.float64)
    for c in range(y.shape[1]):
        solution = class_solutions(x[:, c], y[:, c], b[:, c], (penalty,))[penalty]
        weights[c] = margin * solution[:, 0] + solution[:, 1]
    return {"weight": weights, "valid": valid, "positive_counts": positives,
            "negative_counts": negatives, "margin": margin, "penalty": penalty}


def predict_pairwise(state, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != state["weight"].shape:
        raise ValueError("Query features differ from fitted class/dimension")
    residual = np.einsum("ncd,cd->nc", x, state["weight"])
    residual[:, ~state["valid"]] = 0.
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite residual prediction")
    return residual


def grouped_oof_grid(features, labels, baseline, groups, margins=MARGINS,
                     penalties=PENALTIES, deadline=None, stop_file=None,
                     progress_callback=None):
    """Fit all predeclared small-grid models without any test arrays."""
    x, y, b = validate_support(features, labels, baseline)
    groups = np.asarray(groups, dtype=str)
    if groups.shape != (len(y),) or np.any(groups == "") or len(np.unique(groups)) < 2:
        raise ValueError("Need at least two nonempty support groups")
    output = {(margin, penalty): np.zeros_like(b)
              for margin in margins for penalty in penalties}
    kernels = [x[:, c] @ x[:, c].T for c in range(y.shape[1])]
    folds, missing_rows = [], np.zeros(y.shape[1], dtype=int)
    all_groups = np.unique(groups)
    for index, group in enumerate(all_groups):
        check_limits(deadline, stop_file)
        held, fit = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
        positive = y[fit].sum(axis=0).astype(int)
        negative = len(fit) - positive
        valid = (positive > 0) & (negative > 0)
        missing_rows += (~valid).astype(int) * len(held)
        for c in range(y.shape[1]):
            check_limits(deadline, stop_file)
            if not valid[c]:
                continue
            solutions = class_solutions(
                x[fit, c], y[fit, c], b[fit, c], penalties,
                gram=kernels[c][np.ix_(fit, fit)])
            for penalty, directions in solutions.items():
                parts = x[held, c] @ directions
                for margin in margins:
                    output[margin, penalty][held, c] = margin * parts[:, 0] + parts[:, 1]
        folds.append({
            "held_group_sha256": hashlib.sha256(group.encode()).hexdigest(),
            "held_indices": held.tolist(), "fit_indices": fit.tolist(),
            "fit_positive_counts": positive.tolist(), "fit_negative_counts": negative.tolist(),
            "zero_residual_classes": np.flatnonzero(~valid).tolist(),
            "fit_labels_sha256": array_digest(y[fit]),
        })
        if progress_callback is not None:
            progress_callback(index + 1, len(all_groups))
    return output, {"fold_count": len(folds), "folds": folds,
                    "missing_state_rows_per_class": missing_rows.tolist()}


def pairwise_oof_risk(labels, oof_logits, reference_margin=RISK_MARGIN):
    """Whole OOF table, pairs only within one class; never mix class logits."""
    labels, logits = np.asarray(labels), np.asarray(oof_logits, dtype=np.float64)
    if labels.shape != logits.shape or labels.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("Risk needs matching binary labels and finite OOF logits")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Risk labels must be binary")
    losses, counts = [], []
    for c in range(labels.shape[1]):
        positive, negative = logits[labels[:, c] == 1, c], logits[labels[:, c] == 0, c]
        count = len(positive) * len(negative)
        counts.append(count)
        losses.append(None)
        if count:
            difference = positive[:, None] - negative[None, :]
            losses[-1] = float(np.mean(np.maximum(0., reference_margin - difference) ** 2))
    valid = [value for value in losses if value is not None]
    return {"macro_squared_hinge_risk": float(np.mean(valid)) if valid else None,
            "reference_margin": reference_margin, "per_class": losses,
            "pairs_per_class": counts, "defined_classes": len(valid),
            "scope": "All support OOF rows; pairs only within the same class",
            "cross_fold_calibration_caveat": True}


def candidate_order(record):
    config = record["config"]
    risk = record["support_oof_pair_risk"]["macro_squared_hinge_risk"]
    return (-record["selection_score"], float("inf") if risk is None else risk,
            config["strength"], -config["parameter"], config["margin"])


def number_name(value):
    return format(value, ".12g").replace(".", "p").replace("-", "m")


def source_snapshot(output):
    names = [Path(__file__).name, "overnight_linear_residual.py",
             "overnight_cache_methods.py", "i2cl_data.py", "batch_data.py",
             "batch_metrics.py", "asset_paths.py", "python_local.sh", "activate.sh"]
    root = output / "source_snapshot"
    root.mkdir()
    result = {}
    for name in names:
        path = Path(__file__).with_name(name)
        (root / name).write_bytes(path.read_bytes())
        result[str(path.resolve())] = sha256(path)
    return result


def run(args):
    check_limits(args.deadline, args.stop_file)
    cache = FrozenCache(args.cache)
    if args.dataset and args.dataset != cache.dataset:
        raise ValueError("Dataset differs from frozen cache")
    support = cache.support()
    labels = np.asarray(support["labels"], dtype=np.int8)
    baseline = np.asarray(cache.array("support", "baseline_logits"), dtype=np.float64)
    x = normalized_features(cache.array("support", "class_local"), labels.shape[1])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_snapshot(output)
    support_spec = {"name": cache.dataset, "labels": labels, "class_names": support["class_names"]}
    baseline_metrics = compute_metrics(support_spec, baseline)
    config = {
        "version": VERSION, "dataset": cache.dataset, "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "source_sha256": sources, "cache_manifest": str(cache.path),
        "cache_manifest_sha256": sha256(cache.path), "checkpoint": cache.manifest["checkpoint"],
        "support_csv_sha256": cache.metadata["support"]["csv_sha256"],
        "test_csv_sha256": cache.metadata["test"]["csv_sha256"],
        "support_labels_sha256": array_digest(labels),
        "support_groups_sha256": array_digest(np.asarray(support["groups"], dtype=str)),
        "feature": "class_local", "margins": list(MARGINS), "penalties": list(PENALTIES),
        "strengths": list(STRENGTHS), "risk_reference_margin": RISK_MARGIN,
        "objective": "mean_positive_negative_pairs[(baseline_diff + feature_diff*w - margin)^2] + lambda*||w||²",
        "selection": "Whole group-LOO OOF mean(macro AUROC,macro trapezoidal AUPRC); exact ties: smaller common-margin squared-hinge OOF pair risk, smaller strength, larger lambda, smaller margin.",
        "risk_caveat": "Pairs within each class can use scores from different fitted folds. Fold-specific calibration affects risk and rank selection.",
        "class_fallback": "Zero residual for any fit subset lacking positives or negatives",
        "test_labels_used_for_fit_or_selection": False, "exploratory": True,
        "adapted_maps_available": False,
        "map_limitation": "This score residual does not modify original RadZero similarity maps; they do not explain the added residual.",
    }
    write_json(output / "run_config.json", config)
    try:
        def progress(done, total):
            if done == 1 or done % 10 == 0 or done == total:
                write_json(output / "status.json", {"status": "support_group_cv",
                                                   "completed_folds": done, "fold_count": total})
                print(f"{cache.dataset}: pairwise OOF {done}/{total}", flush=True)

        residuals, audit = grouped_oof_grid(
            x, labels, baseline, support["groups"], deadline=args.deadline,
            stop_file=args.stop_file, progress_callback=progress)
        write_json(output / "fold_audit.json", audit)
        records = []
        for (margin, penalty), residual in residuals.items():
            check_limits(args.deadline, args.stop_file)
            stem = f"margin{number_name(margin)}_lambda{number_name(penalty)}"
            np.save(output / f"oof_residual_{stem}.npy", residual, allow_pickle=False)
            for strength in STRENGTHS:
                scores = baseline + strength * residual
                metrics = compute_metrics(support_spec, scores)
                records.append({
                    "candidate_id": f"class_local_pairwise_{stem}_strength{number_name(strength)}",
                    "config": {"method": "pairwise_ridge", "feature": "class_local",
                               "margin": margin, "parameter": penalty, "strength": strength},
                    "support_oof_metrics": metrics, "support_oof_delta": metric_delta(baseline_metrics, metrics),
                    "selection_score": metric_score(metrics),
                    "support_oof_pair_risk": pairwise_oof_risk(labels, scores),
                    "oof_residual_file": f"oof_residual_{stem}.npy",
                })
        records.sort(key=candidate_order)
        selected = records[0]
        write_json(output / "candidates.json", records)
        write_json(output / "selection.json", {
            "selected_before_test_labels_loaded": True,
            "selection_time": datetime.now(timezone.utc).isoformat(),
            "selection_rule": config["selection"], "pair_risk_caveat": config["risk_caveat"],
            "baseline_support_metrics": baseline_metrics, "selected": [selected],
            "all_candidate_count": len(records), "candidates_sha256": sha256(output / "candidates.json"),
            "support_labels_sha256": array_digest(labels), "zero_strength_retained": True,
        })
        selection_hash = sha256(output / "selection.json")
        choice = selected["config"]
        state = fit_pairwise(x, labels, baseline, choice["margin"], choice["parameter"])
        destination = output / selected["candidate_id"]
        destination.mkdir()
        np.savez(destination / "pairwise_residual.npz", **state,
                 strength=choice["strength"], support_identity=np.asarray(cache.manifest["support_identity"]),
                 support_labels_sha256=np.asarray(array_digest(labels)))
        # Query features and baseline scores have no labels. All model fitting
        # and support-only selection above are complete before test prediction.
        test_x = cache.array("test", "class_local")
        test_base = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float64)
        predictions = test_base.copy()
        for start in range(0, len(test_x), args.chunk_size):
            check_limits(args.deadline, args.stop_file)
            stop = min(start + args.chunk_size, len(test_x))
            if choice["strength"] != 0:
                query = normalized_features(test_x[start:stop], labels.shape[1])
                predictions[start:stop] += choice["strength"] * predict_pairwise(state, query)
        np.save(destination / "predictions.npy", predictions, allow_pickle=False)
        if choice["strength"] == 0:
            np.testing.assert_array_equal(predictions, test_base)
        write_json(output / "status.json", {"status": "predictions_fixed_before_test_metrics",
                                           "selection_sha256": selection_hash,
                                           "prediction_sha256": sha256(destination / "predictions.npy")})
        check_limits(args.deadline, args.stop_file)
        # Sole reader of test labels, after selected fit/predictions are saved.
        evaluation = evaluate_test_only(cache.split_dir, cache.dataset, predictions, test_base,
                                        cache.metadata["test"], destination)
        if sha256(output / "selection.json") != selection_hash:
            raise ValueError("Selection changed during evaluation")
        record = {**selected, **evaluation, "selection_sha256": selection_hash,
                  "prediction_sha256": sha256(destination / "predictions.npy"),
                  "fit_sha256": sha256(destination / "pairwise_residual.npz")}
        write_json(output / "summary.json", {"status": "complete", "dataset": cache.dataset,
                                            "results": [record], "selection_sha256": selection_hash,
                                            "test_labels_used_for_selection": False, "exploratory": True})
        write_json(output / "status.json", {"status": "complete", "dataset": cache.dataset})
        print(json.dumps({"dataset": cache.dataset, "candidate": choice, "delta": evaluation["delta"]}), flush=True)
        return record
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "stopped" if isinstance(error, StopRequested) else "failed",
            "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--deadline", default=HARD_DEADLINE_TEXT)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline.timestamp() > HARD_DEADLINE:
        parser.error("Deadline must have timezone and not exceed authorized 10:00")
    args.deadline = deadline.timestamp()
    if args.chunk_size < 1:
        parser.error("chunk-size must be positive")
    check_limits(args.deadline, args.stop_file)
    def alarm(_signum, _frame):
        raise StopRequested("Hard deadline alarm")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time()))
    try:
        with threadpool_limits(limits=4):
            run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)


if __name__ == "__main__":
    main()
