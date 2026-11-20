#!/usr/bin/env python3
"""Independent CPU audit after both complete neighbor fits and final statistics.

Does not import the experiment's fitting, metric, bootstrap, or summary code.
It reads test labels only after checking all complete fitting commitments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

MODELS = ("BiomedCLIP", "AlphaRAD")
SEEDS = (42, 43, 44, 45, 46)
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SINGLE = ("original", "query", "prototype", "prototype_mix", "tip_adapter", "gda",
          "simnl", "project_mix", "linear_probe")
METHODS = SINGLE + tuple(x + "_view" for x in SINGLE)
METRICS = ("auroc", "auprc", "average_precision")
SCOPES = TASKS + ("macro_three_tasks",)
TOL = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_json(reference):
    require(sha(reference["path"]) == reference["sha256"], "Frozen JSON changed: " + reference["path"])
    return json.loads(Path(reference["path"]).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def close(actual, expected, message, tolerance=TOL):
    if actual is None or expected is None:
        require(actual is expected, message)
        return 0.
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True), message)
    difference = np.abs(a - b)
    return float(np.max(difference[np.isfinite(difference)])) if np.isfinite(difference).any() else 0.


def sklearn_metrics(labels, scores, weights=None):
    if weights is not None:
        active = np.asarray(weights) > 0
        labels, scores, weights = labels[active], scores[active], weights[active]
    if len(np.unique(labels)) < 2:
        return np.full(3, np.nan)
    precision, recall, _ = precision_recall_curve(labels, scores, sample_weight=weights)
    return np.asarray([roc_auc_score(labels, scores, sample_weight=weights), auc(recall, precision),
                       average_precision_score(labels, scores, sample_weight=weights)])


def unit(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def independent_cmp(x, y, positive, negative, test, lam, gamma):
    """Direct dense equations, independent of thin-SVD implementation."""
    means = np.stack((x[y == 0].mean(0), x[y == 1].mean(0)))
    texts = np.stack((negative, positive))
    u, singular, _ = np.linalg.svd(texts.T, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(singular ** 2) / np.sum(singular ** 2), .99) + 1)
    projector = u[:, :k] @ u[:, :k].T
    mixed = float(lam) * (means @ projector) + (1.0 - float(lam)) * texts
    semantic = unit(test @ projector) @ unit(mixed).T
    residual = x - means[y]
    covariance = np.cov(residual.T, ddof=1)
    matrix = (len(x) - 1) * covariance + np.trace(covariance) * np.eye(x.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    require(np.all(eigenvalues > 0), "Undefined dense inverse covariance")
    whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    image = unit(test @ whitening) @ unit(means @ whitening).T
    score = semantic + float(gamma) * image
    return score[:, 1] - score[:, 0], k


def audit_support_and_methods(fitted, bundle):
    rows, folds = bundle["rows"], bundle["folds"]
    require(fitted["support_images"] == [r["image_path"] for r in rows], "Support image identity differs")
    require(fitted["support_patient_ids"] == [r["patient_id"] for r in rows], "Support patient identity differs")
    require(fitted["support_labels"] == [r["label"] for r in rows], "Support labels differ")
    require(fitted["folds"] == folds and len(folds) == 5, "Shared support folds differ")
    require(len(rows) == 10 and len(set(fitted["support_patient_ids"])) == 10, "Not ten support patients")
    require(sum(fitted["support_labels"]) == 5, "Support class balance differs")
    coverage = []
    for fold in folds:
        tr, va = fold["train"], fold["validation"]
        require(len(tr) == 8 and len(va) == 2 and sorted(tr + va) == list(range(10)), "Invalid OOF partition")
        require(sum(rows[i]["label"] for i in va) == 1, "Invalid held-out class balance")
        require(not {rows[i]["patient_id"] for i in tr} & {rows[i]["patient_id"] for i in va}, "Patient crosses fold")
        coverage.extend(va)
    require(sorted(coverage) == list(range(10)), "Support sample missing/repeated in OOF coverage")
    require(set(fitted["methods"]) == set(SINGLE[2:]), "Missing comparator fits")
    expected_fold_hash = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()
    for method, info in fitted["methods"].items():
        require(info["dual_view_parameters_identical"] is True and info["test_labels_received"] is False,
                "Dual view refit or label access: " + method)
        require(info["support_views"] == info["validation_views"] == 1 and info["test_views"] == 2,
                "View budget differs: " + method)
        if method in ("prototype", "prototype_mix", "tip_adapter", "gda"):
            require(info["folds_sha256"] == expected_fold_hash, "Classic method fold hash differs")
        elif method == "simnl":
            require(info["folds"] == folds, "SimNL folds differ")
        else:
            actual = [{"train": f["train_indices"], "validation": f["validation_indices"]} for f in info["folds"]]
            require(actual == folds, "Neighbor method folds differ")
            candidates = info["candidates"]
            require(len(candidates) == (100 if method == "project_mix" else 9), "Candidate grid truncated")
            best = max(candidates, key=lambda candidate: candidate["oof_macro_auprc"])
            params = ("lambda", "gamma", "oof_macro_auprc") if method == "project_mix" else ("C", "oof_macro_auprc")
            for parameter in params:
                close(info["selected"][parameter], best[parameter], "Selected candidate is not support OOF argmax", 0)
    return 7


def audit_scores_from_cache(cache, fits, predictions):
    pool = arrays(cache / "support_pool.npz")
    features = arrays(cache / "test_features.npz")
    texts = arrays(cache / "queries.npz")
    lookup = {str(Path(path).resolve()): i for i, path in enumerate(pool["image_paths"].tolist())}
    sample = np.arange(16)
    test = np.concatenate((features["features_identity"][sample], features["features_center"][sample])).astype(np.float64)
    max_errors, ranks = {}, []
    for j, task in enumerate(TASKS):
        fitted = fits[task]
        indices = [lookup[str(Path(path).resolve())] for path in fitted["support_images"]]
        x = pool["features"][indices].astype(np.float64)
        y = np.asarray(fitted["support_labels"])
        info = fitted["methods"]["project_mix"]
        selected = info["selected"]
        scores, rank = independent_cmp(x, y, texts["positive"][j].astype(np.float64),
                                      texts["negative"][j].astype(np.float64), test,
                                      selected["lambda"], selected["gamma"])
        ranks.append(rank)
        require(rank == info["full_support_fits"][0]["text_rank"], "Recorded SVD rank differs")
        max_errors[task + "/project_mix"] = close(predictions["project_mix"][sample, j], scores[:16], "P&M identity formula differs")
        max_errors[task + "/project_mix_view"] = close(predictions["project_mix_view"][sample, j],
                                                       (scores[:16] + scores[16:]) * .5, "P&M dual formula differs")
        info = fitted["methods"]["linear_probe"]
        coefficient, intercept = np.asarray(info["coefficients"])[0], info["intercepts"][0]
        scores = test @ coefficient + intercept
        max_errors[task + "/linear_probe"] = close(predictions["linear_probe"][sample, j], scores[:16], "LP identity formula differs")
        max_errors[task + "/linear_probe_view"] = close(predictions["linear_probe_view"][sample, j],
                                                        (scores[:16] + scores[16:]) * .5, "LP dual formula differs")
    return dict(max_absolute_error=max(max_errors.values()), comparisons=len(max_errors),
                deterministic_test_rows=16, text_ranks=ranks)


def expected_comparisons():
    definitions = []
    for suffix in ("", "_view"):
        for low in SINGLE[2:] + ("original",):
            definitions.append(("query" + suffix + "_minus_" + low + suffix, "query" + suffix, low + suffix))
    definitions.extend((method + "_view_minus_" + method, method + "_view", method) for method in SINGLE)
    return definitions


def interval(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None, 0
    # Identical percentile definition; independent NumPy call on saved draws.
    low, high = np.quantile(finite, [(1 - .95) / 2, 1 - (1 - .95) / 2], method="linear")
    return float(100 * low), float(100 * high), len(finite)


def audit_rows(summary, point_by_model, draws_by_model):
    expected_counts = {"mean_sd_results": 432, "paired_comparisons": 600,
                       "per_seed_results": 2160, "per_seed_paired_comparisons": 3000}
    for key, count in expected_counts.items():
        require(len(summary[key]) == count, "Summary row count differs: " + key)
    max_error, checked = 0., 0
    for key in expected_counts:
        seen = set()
        for row in summary[key]:
            model, task, metric = row["model"], row["task"], row["metric"]
            require(model in MODELS and task in SCOPES and metric in METRICS, "Unknown result axis")
            ti, mi = SCOPES.index(task), METRICS.index(metric)
            points, draws = point_by_model[model], draws_by_model[model]
            is_pair, is_seed = "paired" in key, key.startswith("per_seed")
            if is_pair:
                definition = next((d for d in expected_comparisons() if d[0] == row["comparison"]), None)
                require(definition is not None, "Unplanned paired comparison")
                _, high, low = definition
                require(row["high_branch"] == high and row["low_branch"] == low, "Pair directions changed")
                hi, lo = METHODS.index(high), METHODS.index(low)
                value = points[:, hi, ti, mi] - points[:, lo, ti, mi]
                samples = draws[:, :, hi, ti, mi] - draws[:, :, lo, ti, mi]
                label = row["comparison"]
            else:
                bi = METHODS.index(row["branch"])
                value, samples = points[:, bi, ti, mi], draws[:, :, bi, ti, mi]
                label = row["branch"]
            identifier = (model, task, metric, label, row.get("support_seed"))
            require(identifier not in seen, "Duplicate result row")
            seen.add(identifier)
            if is_seed:
                si = SEEDS.index(row["support_seed"])
                scalar, replicate = value[si], samples[si]
                field = "delta_pp" if is_pair else "metric_percent"
            else:
                scalar, replicate = value.mean(), samples.mean(axis=0)
                field = "mean_delta_pp" if is_pair else "mean_percent"
                sd_field = "support_seed_delta_sd_pp" if is_pair else "support_seed_sd_pp"
                max_error = max(max_error, close(row[sd_field], 100 * value.std(ddof=1), "Support SD differs"))
                if is_pair:
                    require(row["positive_gain_seed_count"] == int(np.sum(value > 0)) and
                            row["negative_gain_seed_count"] == int(np.sum(value < 0)), "Seed gain sign count differs")
            max_error = max(max_error, close(row[field], 100 * scalar, "Summary point estimate differs"))
            if not is_seed or is_pair:
                low, high, valid = interval(replicate)
                for tail, expected in (("low", low), ("high", high)):
                    suffix = "pp" if is_pair else "percent"
                    max_error = max(max_error, close(row[f"ci_{tail}_{suffix}"], expected, "Percentile interval differs"))
                require(row["valid_replicates"] == valid and row["invalid_replicates"] == len(replicate) - valid,
                        "Invalid bootstrap count differs")
            checked += 1
    return dict(rows_checked=checked, task_method_means=108, macro_method_means=36,
                metric_columns=3, maximum_absolute_error_percent_or_pp=max_error)


def audit(run, protocol_path, draw_checks):
    # Completion gate before the test-label asset is opened or any metric runs.
    catalog = json.loads(protocol_path.read_text())
    summary = json.loads((run / "RESULTS.json").read_text())
    require(summary["status"] == "complete", "Final statistics are not complete")
    require(summary["task_seed_branch_results"] == 540, "Complete result scope differs")
    plan = json.loads((run / "FROZEN_RUN_PLAN.json").read_text())
    require(plan["all_results_retained"] is True and plan["test_labels_opened_for_fitting"] is False,
            "Run plan scope or label isolation differs")
    sources = {str(protocol_path): sha(protocol_path), str(run / "RESULTS.json"): sha(run / "RESULTS.json")}
    def check_source(path, expected):
        if str(path) not in sources:
            sources[str(path)] = sha(path)
        require(sources[str(path)] == expected, "Source or artifact changed: " + str(path))
    for path, expected in plan["sources"].items():
        check_source(path, expected)
    assets = catalog["neighbor_assets"]
    identity, supports = checked_json(assets["test_identity"]), checked_json(assets["support_folds"])["seeds"]
    require(identity["task_keys"] == list(TASKS), "Test task order differs")
    patients = np.asarray(identity["patient_ids"])
    require(len(patients) == 7407 and len(np.unique(patients)) == 2470, "Test patient scope differs")
    all_predictions, all_fits, committed_times = {}, {}, []
    method_fits_checked = 0
    for model in MODELS:
        folder, cache = run / model / "evaluation", run / model / "cache"
        status = json.loads((folder / "status.json").read_text())
        require(status["status"] == "complete" and status["completed_seeds"] == list(SEEDS) and
                status["test_labels_opened"] is False and status["task_seed_fits"] == 15,
                "All fifteen fits must be complete before any test labels are opened")
        require(status["protocol_sha256"] == sha(protocol_path), "Fit protocol hash differs")
        check_source(cache / "result.json", status["cache_result_sha256"])
        cache_record = json.loads((cache / "result.json").read_text())
        require(cache_record["status"] == "complete" and cache_record["rows"] == 7407,
                "Frozen feature cache incomplete")
        for relative, digest in cache_record["artifact_sha256"].items():
            check_source(cache / relative, digest)
        all_predictions[model], all_fits[model] = {}, {}
        for seed in SEEDS:
            seed_folder = folder / f"seed_{seed}"
            receipt = json.loads((seed_folder / "fit_commit.json").read_text())
            require(receipt["status"] == "committed" and receipt["test_labels_opened"] is False and
                    receipt["seed"] == seed and receipt["model"] == model, "Invalid fit commitment")
            for path, digest in receipt["sources"].items():
                check_source(path, digest)
            for name, digest_key in (("predictions.npz", "predictions_sha256"), ("fit_info.json", "fit_info_sha256")):
                check_source(seed_folder / name, receipt[digest_key])
            fits = json.loads((seed_folder / "fit_info.json").read_text())
            for task in TASKS:
                bundle = supports[str(seed)][task]
                require(not set(bundle["groups"]) & set(patients), "Support and test patients overlap")
                method_fits_checked += audit_support_and_methods(fits[task], bundle)
            predictions = arrays(seed_folder / "predictions.npz")
            require(set(predictions) == set(METHODS) | {"image_paths", "patient_ids", "task_keys", "class_names", "seed"},
                    "Missing or extra prediction branches")
            for key in ("image_paths", "patient_ids", "task_keys", "class_names"):
                require(np.array_equal(predictions[key], identity[key]), "Prediction identities differ: " + key)
            require(int(predictions["seed"]) == seed, "Prediction seed differs")
            native = arrays(cache / f"seed_{seed}" / "native_predictions.npz")
            for branch in METHODS:
                require(predictions[branch].shape == (7407, 3) and np.isfinite(predictions[branch]).all(), "Invalid prediction scores")
                if branch in ("original", "query", "original_view", "query_view"):
                    require(np.array_equal(predictions[branch], native[branch]), "Native score changed during evaluation")
                if seed != SEEDS[0] and branch in ("original", "original_view"):
                    require(np.array_equal(predictions[branch], all_predictions[model][SEEDS[0]][branch]), "Original changes across supports")
            all_predictions[model][seed], all_fits[model][seed] = predictions, fits
            committed_times.append(datetime.fromisoformat(receipt["committed_at"]))
    require(max(committed_times) <= datetime.fromisoformat(summary["test_labels_opened_at_utc"]),
            "Statistics opened test labels before the last fitting commitment")
    labels_asset = checked_json(assets["test_labels"])
    require(labels_asset["image_paths"] == identity["image_paths"], "Label image order differs")
    labels = np.asarray(labels_asset["labels"])
    require(labels.shape == (7407, 3) and np.isin(labels, (0, 1)).all(), "Invalid test labels")
    plan_values = arrays(run / "patient_draws.npz")
    counts, unique = plan_values["counts"], plan_values["patient_ids"]
    require(np.array_equal(unique, np.unique(patients)) and counts.shape == (2000, 2470), "Patient draw axes differ")
    require(np.all(counts.sum(1) == 2470) and counts.dtype.kind in "iu", "Invalid patient draw multiplicities")
    rng = np.random.Generator(np.random.PCG64(int(plan_values["seed"])))
    for i in range(2000):
        draw = rng.integers(0, 2470, size=2470)
        require(np.array_equal(counts[i], np.bincount(draw, minlength=2470)), "Patient random draw differs")
    draw_ids = np.unique(np.linspace(0, 1999, draw_checks, dtype=int))
    require(len(draw_ids) >= 20, "At least twenty distinct patient draws must be checked")
    inverse = np.searchsorted(unique, patients)
    point_by_model, draws_by_model, model_reports = {}, {}, {}
    for model in MODELS:
        print("AUDIT", model, "all points, patient draws and fit/view formulas", flush=True)
        boot = arrays(run / "bootstrap" / model / "bootstrap_metrics.npz")
        names = boot["branch_names"].tolist()
        require(boot["task_names"].tolist() == list(TASKS) and boot["metric_names"].tolist() == list(METRICS), "Bootstrap task/metric axes differ")
        require(boot["point_metrics"].shape == (82, 3, 3) and boot["replicate_metrics"].shape == (2000, 82, 3, 3), "Bootstrap shape differs")
        points = np.empty((5, 18, 3, 3))
        draws = np.empty((5, 2000, 18, 3, 3))
        point_error, draw_error = 0., 0.
        verified_draw_branches, formula_audits = set(), []
        for si, seed in enumerate(SEEDS):
            pred = all_predictions[model][seed]
            formula_audits.append(audit_scores_from_cache(run / model / "cache", all_fits[model][seed], pred))
            for bi, method in enumerate(METHODS):
                name = method if method in ("original", "original_view") else f"seed_{seed}__{method}"
                cached_index = names.index(name)
                for ti in range(3):
                    points[si, bi, ti] = sklearn_metrics(labels[:, ti], pred[method][:, ti])
                point_error = max(point_error, close(points[si, bi], boot["point_metrics"][cached_index], "Independent point metrics differ"))
                draws[si, :, bi] = boot["replicate_metrics"][:, cached_index]
                if name not in verified_draw_branches:
                    for di in draw_ids:
                        weights = counts[di, inverse]
                        for ti in range(3):
                            expected = sklearn_metrics(labels[:, ti], pred[method][:, ti], weights)
                            draw_error = max(draw_error, close(expected, boot["replicate_metrics"][di, cached_index, ti], "Independent patient metrics differ"))
                    verified_draw_branches.add(name)
        require(set(names) == verified_draw_branches, "Unverified bootstrap branches remain")
        point_by_model[model] = np.concatenate((points, points.mean(2, keepdims=True)), axis=2)
        draws_by_model[model] = np.concatenate((draws, draws.mean(3, keepdims=True)), axis=3)
        model_reports[model] = dict(task_seed_branch_points=270, unique_score_vectors=len(names) * 3,
            patient_draws_per_score=len(draw_ids), point_max_absolute_error=point_error,
            patient_draw_max_absolute_error=draw_error, independent_formula_audits=formula_audits)
    rows_report = audit_rows(summary, point_by_model, draws_by_model)
    for path, digest in sources.items():
        require(sha(path) == digest, "An audited input changed while auditing: " + path)
    return dict(status="passed", checked_at_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(__file__), protocol_sha256=sha(protocol_path),
        results_sha256=sha(run / "RESULTS.json"), test_labels_read_after_all_fits_committed=True,
        task_seed_branch_results=540, comparator_task_seed_fits_checked=method_fits_checked,
        selected_candidate_rule_checked="first exact maximal support OOF AUPRC for P&M and LP",
        support_folds_and_single_dual_fit_metadata_match=True, native_predictions_unchanged=True,
        draw_indices=draw_ids.tolist(), independent_engine="sklearn metrics; NumPy direct dense equations and quantiles",
        all_metric_points_independently_recomputed=True, all_intervals_recomputed_from_saved_paired_draw_metrics=True,
        bootstrap_scope="At least twenty shared patient draws independently recomputed for every unique score vector; all 2000 draw multiplicities regenerated",
        limitations="SimNL dual fit reuse checked through immutable fit metadata and wrapper source, not an independently refitted SimNL model",
        models=model_reports, summary_rows=rows_report, source_files_sha256=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--draw-checks", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(20 <= args.draw_checks <= 2000, "draw-checks must be between 20 and 2000")
    run, protocol = args.run.resolve(), args.protocol.resolve()
    output = args.output or run / "INDEPENDENT_AUDIT.json"
    try:
        report = audit(run, protocol, args.draw_checks)
    except Exception as error:
        report = dict(status="failed", error=str(error), traceback=traceback.format_exc(),
                      checked_at_utc=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "task_seed_branch_results", "summary_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
