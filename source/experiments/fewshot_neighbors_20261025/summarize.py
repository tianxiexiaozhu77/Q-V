#!/usr/bin/env python3
"""Score all committed same-support/same-view neighbors with paired patients.

The label-free protocol, both complete fitting receipts and every prediction
archive are verified before the separate frozen test-label asset is opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from prepare_protocol import (HERE, STABILITY, MODELS, TASKS, SEEDS, METHODS, SINGLE_METHODS,
                              COMPETITORS, comparison_definitions, load_protocol, sha)


METRICS = ("auroc", "auprc", "average_precision")
SCOPES = TASKS + ("macro_three_tasks",)
DISPLAY = dict(zip(SCOPES, ("CSLV", "CDLV", "CCOM", "three-task macro")))
IDENTITY_KEYS = {"image_paths", "patient_ids", "task_keys", "class_names", "seed"}
FIXED_BRANCHES = ("original", "original_view")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_module():
    path = STABILITY / "cluster_bootstrap.py"
    spec = importlib.util.spec_from_file_location("neighbors_patient_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = bootstrap_module()


def checked_hash(path, expected, sources):
    path = Path(path).resolve()
    actual = sources.get(str(path))
    if actual is None:
        actual = sha(path)
        sources[str(path)] = actual
    if actual != expected:
        raise ValueError("Frozen source/result changed: " + str(path))


def verify_run_plan(run, protocol):
    path = run / "FROZEN_RUN_PLAN.json"
    plan = json.loads(path.read_text())
    if (plan.get("models") != list(MODELS) or plan.get("seeds") != list(SEEDS) or
            plan.get("tasks") != list(TASKS) or plan.get("method_conditions") != len(METHODS) or
            plan.get("task_seed_branch_results") != 540 or plan.get("all_results_retained") is not True or
            plan.get("test_labels_opened_for_fitting") is not False or
            Path(plan.get("protocol", "")).resolve() != protocol["path"]):
        raise ValueError("Frozen run scope differs from complete neighbor comparison")
    sources = {str(path): sha(path)}
    if plan["sources"].get(str(protocol["path"])) != sha(protocol["path"]):
        raise ValueError("Run plan protocol SHA256 mismatch")
    for source, digest in plan["sources"].items():
        checked_hash(source, digest, sources)
    return dict(status="verified", path=str(path), sha256=sha(path), sources=sources,
                models=list(MODELS), methods=list(METHODS), seeds=list(SEEDS), tasks=list(TASKS))


def verify_predictions(values, protocol, seed):
    if set(values) != IDENTITY_KEYS | set(METHODS):
        raise ValueError("Prediction archive must contain exactly identities and eighteen unlabeled methods")
    identity = protocol["test_identity"]
    for key in ("image_paths", "patient_ids", "task_keys", "class_names"):
        if not np.array_equal(values[key], np.asarray(identity[key])):
            raise ValueError("Frozen test identity/order differs: " + key)
    scalar = np.asarray(values["seed"])
    if scalar.shape != () or scalar.dtype.kind not in "iu" or int(scalar) != seed:
        raise ValueError("Prediction support seed differs")
    for method in METHODS:
        if values[method].shape != (7407, 3) or not np.isfinite(values[method]).all():
            raise ValueError("Invalid native scores: " + method)


def verify_all_fits(run, protocol):
    """This function must complete for both models before loading test labels."""
    all_scores, audits, sources = {}, {}, {}
    for model in MODELS:
        folder = run / model / "evaluation"
        cache = run / model / "cache"
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text())
        if (status.get("status") != "complete" or status.get("model") != model or
                status.get("completed_seeds") != list(SEEDS) or status.get("task_seed_fits") != 15 or
                status.get("branch_count") != 18 or status.get("task_seed_branch_results") != 270 or
                status.get("test_labels_opened") is not False or status.get("protocol_sha256") != sha(protocol["path"])):
            raise ValueError("All fifteen task fits must be committed for " + model)
        sources[str(status_path)] = sha(status_path)
        cache_result_path = cache / "result.json"
        checked_hash(cache_result_path, status["cache_result_sha256"], sources)
        cache_result = json.loads(cache_result_path.read_text())
        if (cache_result.get("status") != "complete" or cache_result.get("model") != model or
                cache_result.get("rows") != 7407 or cache_result.get("protocol_sha256") != sha(protocol["path"]) or
                cache_result.get("test_metrics_computed") is not False):
            raise ValueError("Incomplete frozen feature cache")
        for relative, digest in cache_result["artifact_sha256"].items():
            artifact = (cache / relative).resolve()
            if not artifact.is_relative_to(cache.resolve()) or "label" in artifact.name.lower():
                raise ValueError("Unexpected labelled or out-of-tree cache artifact")
            checked_hash(artifact, digest, sources)
        for source, digest in status["sources"].items():
            checked_hash(source, digest, sources)
        per_seed, seed_audits = {}, {}
        for seed in SEEDS:
            seed_folder = folder / f"seed_{seed}"
            commit_path = seed_folder / "fit_commit.json"
            commit = json.loads(commit_path.read_text())
            if (commit.get("status") != "committed" or commit.get("model") != model or commit.get("seed") != seed or
                    commit.get("test_labels_opened") is not False or commit.get("protocol_sha256") != sha(protocol["path"]) or
                    commit.get("cache_result_sha256") != sha(cache_result_path)):
                raise ValueError("Invalid fit commitment")
            sources[str(commit_path)] = sha(commit_path)
            for source, digest in commit["sources"].items():
                checked_hash(source, digest, sources)
            fit_path = seed_folder / "fit_info.json"
            prediction_path = seed_folder / "predictions.npz"
            checked_hash(fit_path, commit["fit_info_sha256"], sources)
            checked_hash(prediction_path, commit["predictions_sha256"], sources)
            fits = json.loads(fit_path.read_text())
            if set(fits) != set(TASKS):
                raise ValueError("Incomplete task fits")
            for task in TASKS:
                fitted, expected = fits[task], protocol["support_tasks"][seed][task]
                rows = expected["rows"]
                if (fitted["support_images"] != [r["image_path"] for r in rows] or
                        fitted["support_patient_ids"] != expected["groups"] or
                        fitted["support_labels"] != [r["label"] for r in rows] or fitted["folds"] != expected["folds"] or
                        set(fitted["methods"]) != set(COMPETITORS) or fitted["query"]["eta"] != .3):
                    raise ValueError("Fit used different supports, folds or methods")
                for method in COMPETITORS:
                    info = fitted["methods"][method]
                    if (info.get("dual_view_parameters_identical") is not True or info.get("test_labels_received") is not False or
                            info.get("support_views") != 1 or info.get("validation_views") != 1 or info.get("test_views") != 2):
                        raise ValueError("View budget or support-only selection differs: " + method)
            with np.load(prediction_path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            verify_predictions(values, protocol, seed)
            with np.load(cache / f"seed_{seed}" / "native_predictions.npz", allow_pickle=False) as native:
                for branch in ("original", "query", "original_view", "query_view"):
                    if not np.array_equal(values[branch], native[branch]):
                        raise ValueError("Evaluator changed native reference predictions")
            if per_seed:
                for branch in FIXED_BRANCHES:
                    if not np.array_equal(values[branch], per_seed[SEEDS[0]][branch]):
                        raise ValueError("Support-independent original branch changed across seeds")
            per_seed[seed] = values
            seed_audits[str(seed)] = dict(committed_at=commit["committed_at"], fit_commit_sha256=sha(commit_path),
                                         predictions_sha256=sha(prediction_path), fit_info_sha256=sha(fit_path), tasks=3)
        all_scores[model] = per_seed
        audits[model] = dict(status="verified", task_seed_fits=15, seed_commits=seed_audits,
                             support_folds_match=True, single_fit_reused_for_dual=True, native_predictions_unchanged=True)
    return all_scores, dict(status="verified", verified_at_utc=now(), test_labels_opened=False,
                            complete_models=list(MODELS), total_task_seed_fits=30,
                            task_seed_branch_results=540, models=audits, source_files_sha256=sources)


def bootstrap_model(model, scores_by_seed, labels, patients, plan, output, provenance):
    scores = {branch: scores_by_seed[SEEDS[0]][branch] for branch in FIXED_BRANCHES}
    for seed in SEEDS:
        for branch in METHODS:
            if branch not in FIXED_BRANCHES:
                scores[f"seed_{seed}__{branch}"] = scores_by_seed[seed][branch]
    fingerprint = dict(provenance=provenance, model=model, branch_order=list(scores), task_order=list(TASKS),
                       draw_plan_sha256=plan.sha256, summary_source_sha256=sha(__file__),
                       bootstrap_source_sha256=sha(STABILITY / "cluster_bootstrap.py"))
    metric_path, input_path = output / "bootstrap_metrics.npz", output / "INPUT_FINGERPRINT.json"
    cached = False
    if metric_path.exists() and input_path.exists():
        record = json.loads(input_path.read_text())
        if record["inputs"] == fingerprint and record["metrics_sha256"] == sha(metric_path):
            with np.load(metric_path, allow_pickle=False) as archive:
                if archive["branch_names"].tolist() != list(scores) or archive["task_names"].tolist() != list(TASKS):
                    raise ValueError("Cached bootstrap axis order differs")
                points, draws = archive["point_metrics"], archive["replicate_metrics"]
            cached = True
    if not cached:
        result = bootstrap.paired_cluster_bootstrap(labels, scores, patients, task_names=TASKS, plan=plan)
        points, draws = result.point_metrics, result.replicate_metrics
        bootstrap.save_bootstrap(result, output)
        write_json(input_path, dict(inputs=fingerprint, metrics_sha256=sha(metric_path)))
    max_error = 0.
    for bi, score in enumerate(scores.values()):
        for ti in range(3):
            precision, recall, _ = precision_recall_curve(labels[:, ti], score[:, ti])
            expected = np.array([roc_auc_score(labels[:, ti], score[:, ti]), auc(recall, precision),
                                 average_precision_score(labels[:, ti], score[:, ti])])
            max_error = max(max_error, float(np.max(np.abs(points[bi, ti] - expected))))
    if max_error > 1e-12:
        raise ValueError("Point metrics differ from independent sklearn calculation")
    names = list(scores)
    indices = [[names.index(branch if branch in FIXED_BRANCHES else f"seed_{seed}__{branch}") for branch in METHODS]
               for seed in SEEDS]
    return (np.asarray([points[ids] for ids in indices]), np.asarray([draws[:, ids] for ids in indices]),
            dict(cached=cached, sklearn_point_max_abs_error=max_error, branch_order=names,
                 metrics_sha256=sha(metric_path), input_fingerprint_sha256=sha(input_path)))


def interval(samples):
    values = bootstrap.interval_summary(0., samples)
    ci = values["ci"]
    return dict(ci_low=None if ci is None else 100 * ci[0], ci_high=None if ci is None else 100 * ci[1],
                valid_replicates=values["valid_replicates"], invalid_replicates=values["invalid_replicates"],
                valid_fraction=values["valid_fraction"], confidence_level=values["confidence_level"])


def summarize_model(model, points, draws):
    # [seed, branch, task, metric]; [seed, replicate, branch, task, metric].
    points = np.concatenate((points, points.mean(axis=2, keepdims=True)), axis=2)
    draws = np.concatenate((draws, draws.mean(axis=3, keepdims=True)), axis=3)
    mean_rows, pair_rows, per_seed, per_seed_pairs = [], [], [], []
    means, mean_draws = points.mean(axis=0), draws.mean(axis=0)
    for ti, task in enumerate(SCOPES):
        for bi, branch in enumerate(METHODS):
            base = dict(model=model, task=task, branch=branch,
                        view_condition="dual" if branch.endswith("_view") else "single")
            for mi, metric in enumerate(METRICS):
                ci = interval(mean_draws[:, bi, ti, mi])
                mean_rows.append(dict(**base, metric=metric, mean_percent=float(100 * means[bi, ti, mi]),
                    support_seed_sd_pp=float(100 * points[:, bi, ti, mi].std(ddof=1)),
                    ci_low_percent=ci.pop("ci_low"), ci_high_percent=ci.pop("ci_high"), **ci))
                for si, seed in enumerate(SEEDS):
                    per_seed.append(dict(**base, support_seed=seed, metric=metric,
                                         metric_percent=float(100 * points[si, bi, ti, mi])))
        for definition in comparison_definitions():
            hi, lo = [METHODS.index(definition[key]) for key in ("high", "low")]
            base = dict(model=model, task=task, comparison=definition["name"],
                        high_branch=definition["high"], low_branch=definition["low"],
                        view_condition=definition["view_condition"], comparison_role=definition["role"])
            for mi, metric in enumerate(METRICS):
                delta = points[:, hi, ti, mi] - points[:, lo, ti, mi]
                sample_delta = draws[:, :, hi, ti, mi] - draws[:, :, lo, ti, mi]
                ci = interval(sample_delta.mean(axis=0))
                pair_rows.append(dict(**base, metric=metric, mean_delta_pp=float(100 * delta.mean()),
                    support_seed_delta_sd_pp=float(100 * delta.std(ddof=1)),
                    positive_gain_seed_count=int((delta > 0).sum()), negative_gain_seed_count=int((delta < 0).sum()),
                    ci_low_pp=ci.pop("ci_low"), ci_high_pp=ci.pop("ci_high"), **ci))
                for si, seed in enumerate(SEEDS):
                    seed_ci = interval(sample_delta[si])
                    per_seed_pairs.append(dict(**base, support_seed=seed, metric=metric, delta_pp=float(100 * delta[si]),
                        ci_low_pp=seed_ci.pop("ci_low"), ci_high_pp=seed_ci.pop("ci_high"), **seed_ci))
    return mean_rows, pair_rows, per_seed, per_seed_pairs


def text_report(summary):
    text = ["CheXchoNet same-support and same-view neighbor comparison — all results", "",
            "BiomedCLIP / AlphaRAD; 7407 images / 2470 patients; task supports 5 positive + 5 strict negative patients; seeds 42–46.",
            "Seven competitors: prototype, prototype_mix, Tip-Adapter, GDA, SimNL, Project-and-Mix, linear probe.",
            "Project-and-Mix is an equation-based binary adaptation, not an official executable-code reproduction.",
            "All methods select once from the same original-view support folds. The fitted predictor is reused for both test views.",
            "Compare Q to single-view competitors and Q+V to equally budgeted dual-view competitors; retain all outcomes.",
            "AlphaRAD query methods use class-local support features; score competitors use global projected features.",
            "AUROC / trapezoidal AUPRC are primary; AP is separately recorded. Mean ± SD use metrics of five fits, not ensembled predictions.",
            "95% paired patient-bootstrap intervals condition on fixed supports, no multiplicity correction; this cohort was already evaluated previously.",
            "Scores are percentages; SD and differences are percentage points. No cross-model pooled estimate.", ""]
    means = {(r["model"], r["task"], r["branch"], r["metric"]): r for r in summary["mean_sd_results"]}
    pairs = {(r["model"], r["task"], r["comparison"], r["metric"]): r for r in summary["paired_comparisons"]}
    for model in MODELS:
        text.append(model)
        for task in SCOPES:
            text.append("  " + DISPLAY[task])
            for branch in METHODS:
                cells = []
                for metric in METRICS[:2]:
                    r = means[model, task, branch, metric]
                    cells.append(f"{metric.upper()} {r['mean_percent']:.4f} ± {r['support_seed_sd_pp']:.4f}")
                text.append("    " + branch + ": " + "; ".join(cells))
            for definition in comparison_definitions():
                cells = []
                for metric in METRICS[:2]:
                    r = pairs[model, task, definition["name"], metric]
                    ci = "unavailable" if r["ci_low_pp"] is None else f"[{r['ci_low_pp']:+.4f}, {r['ci_high_pp']:+.4f}]"
                    cells.append(f"{metric.upper()} {r['mean_delta_pp']:+.4f} {ci}; positive {r['positive_gain_seed_count']}/5")
                text.append("    " + definition["name"] + " [" + definition["role"] + "]: " + "; ".join(cells))
        text.append("")
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol_v1/catalog.json")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = load_protocol(args.protocol, include_test_labels=False)
    expected_bootstrap = protocol["catalog"]["experiment"]["bootstrap"]
    if args.replicates != expected_bootstrap["replicates"] or args.seed != expected_bootstrap["seed"]:
        raise ValueError("Bootstrap settings differ from frozen protocol")
    plan_audit = verify_run_plan(run, protocol)
    scores, fit_audit = verify_all_fits(run, protocol)
    fit_audit["run_plan"] = plan_audit
    write_json(run / "FIT_COMPLETION_AUDIT.json", fit_audit)
    # Only now can the scoring-only label asset be opened, after both complete
    # models and all task/support fits were audited and durably recorded.
    labels_opened_at = now()
    labelled = load_protocol(args.protocol, include_test_labels=True)
    labels = np.asarray(labelled["test_labels"], dtype=np.uint8)
    patients = np.asarray(protocol["test_identity"]["patient_ids"])
    plan = bootstrap.make_patient_draw_plan(patients, n_resamples=args.replicates, seed=args.seed)
    bootstrap.save_patient_draw_plan(plan, run / "patient_draws.npz")
    fingerprint = dict(fit_completion_audit_sha256=sha(run / "FIT_COMPLETION_AUDIT.json"),
                       protocol_sha256=sha(args.protocol), test_labels_sha256=labelled["test_labels_source"]["sha256"],
                       run_plan_sha256=plan_audit["sha256"], predictions_and_fits_sha256=fit_audit["source_files_sha256"])
    summary = dict(status="complete", created_at_utc=now(), models=list(MODELS), task_order=list(TASKS),
        support_seeds=list(SEEDS), methods=list(METHODS), competitors=list(COMPETITORS),
        comparison_definitions=comparison_definitions(), test_images=7407, test_patients=2470,
        task_seed_branch_results=540, metric_names=list(METRICS), primary_metrics=list(METRICS[:2]),
        bootstrap_replicates=args.replicates, bootstrap_seed=args.seed, draw_plan_sha256=plan.sha256,
        protocol=str(protocol["path"]), protocol_sha256=sha(protocol["path"]),
        study_type=protocol["catalog"]["experiment"]["study_type"],
        uncertainty_scope=protocol["catalog"]["experiment"]["uncertainty_scope"],
        data_use_history=protocol["catalog"]["experiment"]["data_use_history"],
        summary_policy=protocol["catalog"]["experiment"]["summary_policy"],
        no_cross_model_averaging=True, all_results_retained=True, test_labels_opened_at_utc=labels_opened_at,
        fit_completion_audit_sha256=fingerprint["fit_completion_audit_sha256"], run_plan_audit=plan_audit,
        model_audits={}, source_sha256={str(Path(__file__).resolve()):sha(__file__),
            str(STABILITY / "cluster_bootstrap.py"):sha(STABILITY / "cluster_bootstrap.py")})
    groups = [[], [], [], []]
    for model in MODELS:
        print("BOOTSTRAP", model, flush=True)
        points, draws, audit = bootstrap_model(model, scores[model], labels, patients, plan,
                                                run / "bootstrap" / model, fingerprint)
        summary["model_audits"][model] = audit
        for destination, rows in zip(groups, summarize_model(model, points, draws)):
            destination.extend(rows)
    keys = ("mean_sd_results", "paired_comparisons", "per_seed_results", "per_seed_paired_comparisons")
    filenames = ("MEAN_SD_COMPARISON.csv", "PAIRED_COMPARISONS.csv", "PER_SEED_RESULTS.csv", "PER_SEED_PAIRED_COMPARISONS.csv")
    expected_counts = (432, 600, 2160, 3000)
    for key, filename, rows, expected_count in zip(keys, filenames, groups, expected_counts):
        if len(rows) != expected_count:
            raise ValueError("Incomplete final rows: " + key)
        summary[key] = rows
        write_csv(run / filename, rows)
    changed = [source for source, digest in fit_audit["source_files_sha256"].items() if sha(source) != digest]
    changed += [source for source, digest in plan_audit["sources"].items() if sha(source) != digest]
    if changed:
        raise ValueError("Inputs changed during statistics: " + repr(changed))
    write_json(run / "RESULTS.json", summary)
    (run / "RESULTS.txt").write_text(text_report(summary))
    print(json.dumps(dict(status="complete", models=list(MODELS), task_seed_branch_results=540,
                          means=len(groups[0]), paired_comparisons=len(groups[1])), ensure_ascii=False))


if __name__ == "__main__":
    main()
