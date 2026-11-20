#!/usr/bin/env python3
"""UNEXECUTED PROTOTYPE: support-group OOF query/ridge score fusion.

This draft is not the final campaign method or a validated inference entry.
It was not run on any dataset and has no performance result. Reuse of its old
raw support-token dependency needs an unresolved fresh-encoder consistency
check; a CPU logit anchor cannot substitute for that check.

Only cached support DINO tokens pass through frozen alignment on CPU. Each
support group is aligned once and scored using queries fitted on other groups.
The ridge hyperparameters were independently selected using support OOF by the
wide-ridge experiment. Test predictions combine those existing frozen arrays;
no test encoder is run, and test labels are read only for final evaluation.
The overall campaign remains exploratory after earlier test-set comparisons.
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
import signal
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from batch_metrics import compute_metrics
from infer_batch import MODEL_DIR, checkpoint_identity, load_model
from overnight_cache_methods import (
    DeadlineReached, array_digest, check_deadline, evaluate_test_only, leave_group_out,
    metric_score, normalized_features, sha256, write_json,
)
from overnight_i2cl_variants import load_verified_support_tokens
from overnight_linear_residual import FrozenCache
from overnight_oof_loss_tiebreak import build_bce_folds, candidate_order, weighted_oof_bce
from overnight_query_refinement import (
    HARD_DEADLINE_TEXT, QueryRefinementMatcher, eta_name, fit_query_directions,
    parse_deadline,
)
from train_i2cl import tensor_digest

VERSION = "radzero-support-group-oof-query-ridge-fusion-v1"
FAMILY = "class_local_query_ridge_oof"
ETAS = (0., .01, .03, .1, .3)
RIDGE_FACTORS = (0., .5, 1.)
QUERY_FACTORS = (0., .25, .5, 1.)
SELECTION_RULE = (
    "Fixed support-OOF-selected class_local wide ridge; fusion descending mean "
    "(macro AUROC,macro AUPRC); exact ties ascending fold-fit-weighted OOF BCE, "
    "then smaller RMS OOF logit change, ridge factor, query factor and eta"
)


class StopRequested(RuntimeError):
    pass


def check_control(deadline, stop_files):
    check_deadline(deadline)
    for path in stop_files:
        if Path(path).exists():
            raise StopRequested(f"Campaign STOP file exists: {path}")


def fusion_grid():
    # b=0 makes eta immaterial. Retain one canonical eta=0 for that case.
    return [
        {"ridge_factor": a, "query_factor": b, "eta": eta}
        for a in RIDGE_FACTORS for b in QUERY_FACTORS
        for eta in ((0.,) if b == 0. else ETAS[1:])
    ]


def candidate_id(config):
    return (f"ridge{config['ridge_factor']:g}_query{config['query_factor']:g}"
            f"_{eta_name(config['eta'])}")


def fit_fold_queries(features, labels, original_queries, etas=ETAS):
    """Original query formula on fitting rows only; missing classes stay original."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    original = np.asarray(original_queries)
    if (features.ndim != 3 or labels.shape != features.shape[:2] or not len(labels)
            or original.shape != features.shape[1:] or not np.isin(labels, (0, 1)).all()
            or not np.isfinite(features).all() or not np.isfinite(original).all()):
        raise ValueError("Invalid fitting support features/labels/queries")
    if tuple(etas) != ETAS:
        raise ValueError("This sealed experiment uses the fixed original eta grid")
    positive = labels.sum(axis=0).astype(int)
    negative = len(labels) - positive
    valid = (positive > 0) & (negative > 0)
    queries = np.repeat(original.astype(np.float32)[None], len(etas), axis=0)
    directions = np.zeros_like(original, dtype=np.float32)
    positive_means, negative_means = directions.copy(), directions.copy()
    if valid.any():
        fitted = fit_query_directions(features[:, valid], labels[:, valid], original[valid], etas)
        queries[:, valid] = fitted["queries"]
        directions[valid] = fitted["directions"]
        positive_means[valid] = fitted["positive_means"]
        negative_means[valid] = fitted["negative_means"]
    return {
        "queries": queries, "directions": directions, "valid": valid,
        "positive_means": positive_means, "negative_means": negative_means,
        "positive_counts": positive, "negative_counts": negative,
    }


def query_group_oof(features, labels, baseline, groups, original_queries, tokens,
                    model, matcher, check=lambda: None, progress=None, baseline_atol=2e-5):
    """Fit held-group queries on other groups; frozen per-image alignment is shared."""
    labels, baseline = np.asarray(labels), np.asarray(baseline, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if (groups.shape != (len(labels),) or np.any(groups == "")
            or len(np.unique(groups)) < 2 or baseline.shape != labels.shape
            or len(tokens) != len(labels)):
        raise ValueError("Invalid groups, support tokens or baseline dimensions")
    residuals = {eta_name(eta): np.zeros_like(baseline) for eta in ETAS}
    raw_baseline = np.empty_like(baseline)
    folds, missing = [], np.zeros(labels.shape[1], dtype=int)
    with torch.inference_mode():
        for group in np.unique(groups):
            check()
            held, fit = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
            fitted = fit_fold_queries(features[fit], labels[fit], original_queries)
            aligned = model.align_transformer(tokens[held])
            vision = matcher.prepare_vision(aligned)
            original = torch.from_numpy(np.asarray(original_queries).copy()).float()
            cpu_base = matcher.match_prepared(vision, original)["logits"].cpu().numpy()
            raw_baseline[held] = cpu_base
            error = float(np.abs(cpu_base.astype(float) - baseline[held]).max())
            if error > baseline_atol:
                raise ValueError(f"CPU original query differs from frozen support baseline: {error}")
            for index, eta in enumerate(ETAS):
                check()
                if eta == 0.:
                    continue  # Exactly identity, not a second numerically different subtraction.
                query = torch.from_numpy(fitted["queries"][index].copy())
                values = matcher.match_prepared(vision, query)["logits"].cpu().numpy()
                delta = values.astype(np.float64) - cpu_base.astype(np.float64)
                delta[:, ~fitted["valid"]] = 0.
                residuals[eta_name(eta)][held] = delta
            missing += (~fitted["valid"]).astype(int) * len(held)
            folds.append({
                "held_group_sha256": hashlib.sha256(group.encode()).hexdigest(),
                "held_indices": held.tolist(), "fit_indices": fit.tolist(),
                "fit_positive_counts": fitted["positive_counts"].tolist(),
                "fit_negative_counts": fitted["negative_counts"].tolist(),
                "zero_residual_classes": np.flatnonzero(~fitted["valid"]).tolist(),
                "query_directions_sha256": array_digest(fitted["directions"]),
                "queries_sha256": array_digest(fitted["queries"]),
                "positive_means_sha256": array_digest(fitted["positive_means"]),
                "negative_means_sha256": array_digest(fitted["negative_means"]),
                "cpu_baseline_max_abs_error": error,
            })
            if progress:
                progress(len(folds), len(np.unique(groups)))
    return residuals, {
        "fold_count": len(folds), "folds": folds,
        "missing_state_rows_per_class": missing.tolist(),
        "aligned_support_images": len(labels),
        "cpu_baseline_max_abs_error": float(np.abs(raw_baseline - baseline).max()),
        "cpu_baseline_atol": baseline_atol,
        "numerical_anchor": "cached base + (CPU refined query logit - CPU original query logit)",
        "eta_zero_residual_exact_zero": True,
    }, raw_baseline


def combine_logits(baseline, ridge_delta, query_delta, config):
    baseline = np.asarray(baseline, dtype=np.float64)
    ridge_delta = np.asarray(ridge_delta, dtype=np.float64)
    query_delta = np.asarray(query_delta, dtype=np.float64)
    if (baseline.shape != ridge_delta.shape or baseline.shape != query_delta.shape
            or not all(np.isfinite(x).all() for x in (baseline, ridge_delta, query_delta))):
        raise ValueError("Expected finite, aligned baseline and residual arrays")
    result = baseline.copy()
    if config["ridge_factor"]:
        result += config["ridge_factor"] * ridge_delta
    if config["query_factor"]:
        result += config["query_factor"] * query_delta
    return result


def fusion_order(record):
    bce = record["support_oof_bce"]["weighted_bce"]
    config = record["config"]
    return (-record["selection_score"], math.inf if bce is None else bce,
            record["oof_logit_change_rms"], config["ridge_factor"],
            config["query_factor"], config["eta"], record["candidate_id"])


def support_search(labels, class_names, baseline, ridge_delta, query_deltas, bce_folds):
    """No test inputs or labels accepted by fusion selection."""
    spec = {"name": "support", "labels": labels, "class_names": class_names}
    records, arrays = [], {}
    for config in fusion_grid():
        logits = combine_logits(baseline, ridge_delta, query_deltas[eta_name(config["eta"])], config)
        metrics = compute_metrics(spec, logits)
        bce = weighted_oof_bce(logits, labels, bce_folds)
        record = {
            "candidate_id": candidate_id(config), "family": FAMILY, "config": config,
            "support_oof_metrics": metrics, "support_oof_bce": bce,
            "selection_score": metric_score(metrics),
            "oof_logit_change_rms": float(np.sqrt(np.mean((logits - baseline) ** 2))),
        }
        records.append(record)
        arrays[record["candidate_id"]] = logits
    return sorted(records, key=fusion_order), arrays


def verify_source_hashes(mapping):
    for path, expected in mapping.items():
        if sha256(path) != expected:
            raise ValueError(f"Dependency source changed: {path}")


def read_ridge_reference(directory, cache, labels, groups):
    directory = Path(directory).resolve()
    if json.loads((directory / "status.json").read_text())["status"] != "complete":
        raise ValueError("Wide ridge reference is not complete")
    config = json.loads((directory / "run_config.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    records = json.loads((directory / "candidates.json").read_text())
    if config["cache_manifest_sha256"] != sha256(cache.path):
        raise ValueError("Ridge reference uses a different frozen feature cache")
    verify_source_hashes(config["source_sha256"])
    if (selection["support_labels_sha256"] != array_digest(labels)
            or selection["support_groups_sha256"] != hashlib.sha256(json.dumps(groups).encode()).hexdigest()
            or not selection["selected_before_test_labels_loaded"]):
        raise ValueError("Ridge selection is not bound to this support set")
    choices = [row for row in records if row["config"]["method"] == "ridge"
               and row["config"]["feature"] == "class_local"]
    if not choices:
        raise ValueError("No class-local ridge support candidates")
    chosen = min(choices, key=candidate_order)
    selected = [row for row in selection["candidates"]
                if row["candidate_id"] == chosen["candidate_id"]]
    if len(selected) != 1 or selected[0] != chosen:
        raise ValueError("Stored wide-ridge winner differs from support-only selection")
    return chosen, {
        "directory": str(directory), "candidate_id": chosen["candidate_id"],
        "selection_sha256": sha256(directory / "selection.json"),
        "candidates_sha256": sha256(directory / "candidates.json"),
        "run_config_sha256": sha256(directory / "run_config.json"),
        "source_sha256_verified": config["source_sha256"],
    }


def read_query_reference(directory, cache, labels, features, original_queries):
    directory = Path(directory).resolve()
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if (manifest["status"] != "complete" or manifest["dataset"] != cache.dataset
            or manifest["cache_manifest_sha256"] != sha256(cache.path)
            or manifest["checkpoint"] != cache.manifest["checkpoint"]
            or manifest["support_identity"] != cache.manifest["support_identity"]
            or manifest["test_identity"] != cache.manifest["test_identity"]
            or tuple(manifest["etas"]) != ETAS):
        raise ValueError("Query reference identity or eta grid differs")
    verify_source_hashes(manifest["source_sha256"])
    fit_path = directory / "fit_vectors.npz"
    if sha256(fit_path) != manifest["fit_vectors_sha256"]:
        raise ValueError("Query fitting payload changed")
    fitted = fit_query_directions(features, labels, original_queries, ETAS)
    with np.load(fit_path, allow_pickle=False) as saved:
        for key, value in fitted.items():
            if not np.array_equal(saved[key], value):
                raise ValueError(f"Full-support query refit differs in {key}")
    # Deliberately keep only identity and prediction hashes, never test metrics.
    return {
        "directory": str(directory), "manifest_sha256": sha256(path),
        "fit_vectors_sha256": sha256(fit_path), "full_support_refit_bitwise_equal": True,
        "support_identity": manifest["support_identity"], "test_identity": manifest["test_identity"],
        "predictions": {eta_name(eta): {
            "file": str(directory / eta_name(eta) / "logits.npy"),
            "sha256": manifest["predictions"][eta_name(eta)]["logits_sha256"]}
            for eta in ETAS},
    }


def verified_prediction(path, digest, shape):
    if sha256(path) != digest:
        raise ValueError(f"Saved prediction changed: {path}")
    values = np.load(path, allow_pickle=False)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Prediction shape/order count differs or is nonfinite")
    return values.astype(np.float64)


def final_prediction_from_references(baseline, selected, ridge_reference, query_reference):
    """Use already fixed full-test predictions; no labels or image encoder."""
    path = Path(ridge_reference["directory"]) / ridge_reference["candidate_id"]
    record = json.loads((path / "result.json").read_text())
    if (record["selection_sha256"] != ridge_reference["selection_sha256"]
            or sha256(Path(ridge_reference["directory"]) / "selection.json")
            != ridge_reference["selection_sha256"]):
        raise ValueError("Ridge selected test predictions lost selection provenance")
    ridge = verified_prediction(path / "predictions.npy", record["prediction_sha256"], baseline.shape)
    query_zero = query_reference["predictions"][eta_name(0.)]
    zero = verified_prediction(query_zero["file"], query_zero["sha256"], baseline.shape)
    if not np.array_equal(zero, baseline):
        raise ValueError("Original query full-test baseline differs from frozen baseline")
    query_record = query_reference["predictions"][eta_name(selected["config"]["eta"])]
    query = verified_prediction(query_record["file"], query_record["sha256"], baseline.shape)
    config = selected["config"]
    predictions = combine_logits(baseline, ridge - baseline, query - baseline, config)
    # Preserve the established endpoint arrays exactly, including last-bit addition order.
    if config["ridge_factor"] == 1. and config["query_factor"] == 0.:
        predictions = ridge.copy()
    elif config["ridge_factor"] == 0. and config["query_factor"] == 1.:
        predictions = query.copy()
    return predictions, {
        "ridge_predictions_sha256": record["prediction_sha256"],
        "query_predictions_sha256": query_record["sha256"],
        "query_zero_equals_cached_baseline_bitwise": True,
        "test_encoder_called": False, "test_labels_used": False,
    }


def run(args):
    with threadpool_limits(limits=args.threads):
        torch.set_num_threads(args.threads)
        return _run(args)


def _run(args):
    check = lambda: check_control(args.deadline, args.stop_file)
    check()
    cache = FrozenCache(args.cache)
    if args.dataset and args.dataset != cache.dataset:
        raise ValueError("Dataset/cache mismatch")
    support = cache.support()
    for field in ("ids",):
        if set(cache.metadata["support"][field]) & set(cache.metadata["test"][field]):
            raise ValueError("Support and test identifiers overlap")
    labels = np.asarray(support["labels"], dtype=np.int8)
    features = np.asarray(cache.array("support", "class_local"))
    baseline = np.asarray(cache.array("support", "baseline_logits"), dtype=np.float64)
    query_path = cache.root / "text_features.npy"
    if sha256(query_path) != cache.manifest["auxiliary_files"][query_path.name]["sha256"]:
        raise ValueError("Text features changed")
    original_queries = np.load(query_path, allow_pickle=False)
    ridge, ridge_reference = read_ridge_reference(args.ridge_run, cache, labels, support["groups"])
    query_reference = read_query_reference(
        args.query_run, cache, labels, features, original_queries)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve()] + [
        Path(__file__).with_name(name).resolve() for name in (
            "overnight_cache_methods.py", "overnight_oof_loss_tiebreak.py",
            "overnight_query_refinement.py", "overnight_i2cl_variants.py",
            "overnight_linear_residual.py", "train_i2cl.py", "infer_batch.py",
            "batch_metrics.py", "i2cl_data.py")]
    source_paths += sorted(MODEL_DIR.glob("*.py"))
    plan = {
        "version": VERSION, "family": FAMILY, "dataset": cache.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
        "grid": fusion_grid(), "fixed_ridge": ridge, "selection_rule": SELECTION_RULE,
        "cache_manifest": str(cache.path), "cache_manifest_sha256": sha256(cache.path),
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "checkpoint": cache.manifest["checkpoint"],
        "support_csv_sha256": cache.metadata["support"]["csv_sha256"],
        "test_csv_sha256": cache.metadata["test"]["csv_sha256"],
        "support_identity": cache.manifest["support_identity"],
        "test_identity": cache.manifest["test_identity"],
        "ridge_reference": ridge_reference, "query_reference": query_reference,
        "test_labels_used_for_selection": False, "device": "cpu",
        "numerical_anchor": "cache base + CPU refined-minus-original query score",
        "interpretation": "Exploratory support-selected score fusion; original maps do not explain total residual",
    }
    write_json(output / "plan.json", plan)
    plan_sha = sha256(output / "plan.json")
    write_json(output / "status.json", {"status": "sealed_support_oof", "plan_sha256": plan_sha})
    try:
        check()
        # Recompute the frozen ridge's OOF scores, preserving original fitting folds.
        raw_ridge, ridge_audit = leave_group_out(
            normalized_features(features, labels.shape[1]), labels, baseline, support["groups"],
            "ridge", ridge["config"]["parameter"], args.deadline)
        ridge_delta = ridge["config"]["strength"] * raw_ridge
        bce_folds = build_bce_folds(labels, support["groups"], ridge_audit)
        reproduced = compute_metrics(support, baseline + ridge_delta)
        for key in ("macro_auroc", "macro_auprc"):
            if reproduced[key] != ridge["support_oof_metrics"][key]:
                raise ValueError(f"Rebuilt ridge OOF differs in {key}")
        bce = weighted_oof_bce(baseline + ridge_delta, labels, bce_folds)
        if bce["weighted_bce"] != ridge["support_oof_bce"]["weighted_bce"]:
            raise ValueError("Rebuilt ridge OOF weighted BCE differs")
        check()
        if checkpoint_identity() != cache.manifest["checkpoint"]:
            raise ValueError("Current checkpoint differs from frozen cache")
        for filename, key in (("config.json", "model_config_sha256"),
                              ("preprocessor_config.json", "preprocessor_sha256")):
            if sha256(MODEL_DIR / filename) != cache.manifest[key]:
                raise ValueError("Model configuration/preprocessing differs from cache")
        model, _, _ = load_model(torch.device("cpu"))
        model.float().requires_grad_(False).eval()
        before = tensor_digest(model)
        tokens, token_audit = load_verified_support_tokens(
            args.support_cache, support, cache, torch.device("cpu"), expected_model_digest=before)
        matcher = QueryRefinementMatcher(model, original_queries[None], [0.])

        def progress(done, total):
            check()
            write_json(output / "status.json", {
                "status": "support_query_oof", "folds_done": done, "folds_total": total,
                "updated_at": datetime.now(timezone.utc).isoformat()})
            if done == 1 or done % 10 == 0 or done == total:
                print(f"{cache.dataset}: query support OOF {done}/{total}", flush=True)

        query_delta, query_audit, cpu_baseline = query_group_oof(
            features, labels, baseline, support["groups"], original_queries, tokens,
            model, matcher, check, progress)
        if tensor_digest(model) != before or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Frozen base model was modified")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError("Frozen model unexpectedly received gradients")
        for first, second in zip(ridge_audit["folds"], query_audit["folds"]):
            for key in ("fit_indices", "held_indices", "fit_positive_counts",
                        "fit_negative_counts", "zero_residual_classes"):
                if first[key] != second[key]:
                    raise ValueError("Ridge and query OOF folding differs")
        write_json(output / "support_cache_audit.json", token_audit)
        write_json(output / "fold_audits.json", {
            "query": query_audit, "ridge": ridge_audit, "bce_folds": bce_folds,
            "model_tensor_before_sha256": before, "model_tensor_after_sha256": before,
            "frozen_model_passed": True,
        })
        np.savez_compressed(output / "support_oof_residuals.npz",
                            ridge_delta=ridge_delta, cpu_baseline=cpu_baseline, **query_delta)
        records, oof_logits = support_search(
            labels, support["class_names"], baseline, ridge_delta, query_delta, bce_folds)
        write_json(output / "candidates.json", records)
        selected = records[0]
        selection = {
            "selected_before_test_labels_loaded": True, "selection_time": time.time(),
            "selection_rule": SELECTION_RULE, "plan_sha256": plan_sha,
            "support_labels_sha256": array_digest(labels),
            "support_groups_sha256": array_digest(np.asarray(support["groups"])),
            "support_selected_candidate_id": selected["candidate_id"], "candidates": [selected],
            "candidate_count": len(records), "fixed_ridge": ridge,
            "warning": "Support OOF used for tuning; not an unbiased validation estimate",
        }
        write_json(output / "selection.json", selection)
        selection_sha = sha256(output / "selection.json")
        np.save(output / "selected_support_oof_logits.npy", oof_logits[selected["candidate_id"]],
                allow_pickle=False)
        # All tuning is now sealed. Full test arrays are read only for prediction.
        check()
        if sha256(output / "plan.json") != plan_sha:
            raise ValueError("Sealed plan changed")
        baseline_test = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float64)
        predictions, prediction_audit = final_prediction_from_references(
            baseline_test, selected, ridge_reference, query_reference)
        np.save(output / "predictions.npy", predictions, allow_pickle=False)
        write_json(output / "prediction_audit.json", prediction_audit)
        check()
        if sha256(output / "selection.json") != selection_sha:
            raise ValueError("Selection changed after reading test predictions")
        evaluation = evaluate_test_only(
            cache.split_dir, cache.dataset, predictions, baseline_test, cache.metadata["test"], output)
        result = {
            **selected, **evaluation, "support_selected": True,
            "config": {**selected["config"], "ridge_parameter": ridge["config"]["parameter"],
                       "ridge_strength": ridge["config"]["strength"]},
            "plan_sha256": plan_sha, "selection_sha256": selection_sha,
            "prediction_sha256": sha256(output / "predictions.npy"),
        }
        write_json(output / "result.json", result)
        write_json(output / "summary.json", {
            "dataset": cache.dataset, "version": VERSION, "family": FAMILY,
            "exploratory": True, "selection": selection, "results": [result],
            "test_labels_used_for_fitting_or_selection": False,
        })
        write_json(output / "status.json", {
            "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
    except (DeadlineReached, StopRequested) as error:
        write_json(output / "status.json", {"status": "stopped", "reason": str(error)})
        raise
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "query-run", "ridge-run", "support-cache", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=parse_deadline, default=HARD_DEADLINE_TEXT)
    parser.add_argument("--stop-file", action="append", default=[])
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4:
        parser.error("threads must be 1..4")
    return args


def main():
    args = parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: (_ for _ in ()).throw(
            StopRequested("Received stop signal")))
    signal.signal(signal.SIGALRM, lambda number, frame: (_ for _ in ()).throw(
        DeadlineReached("Hard deadline reached")))
    check_control(args.deadline, args.stop_file)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time()))
    try:
        run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
