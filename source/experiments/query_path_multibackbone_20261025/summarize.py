#!/usr/bin/env python3
"""Paired patient-bootstrap analysis of all frozen multiclass query-path runs.

Strict Q-R and decoder fixed-aggregation Q-F contrasts remain distinct. Metrics
are computed per support seed, then averaged; predictions are never ensembled.
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

from prepare_protocol import ROOT, STABILITY, TASKS, SEEDS, MODEL_CONTROLS, sha, write_json, verify_frozen, inherited_loader


def load_bootstrap_module():
    path = STABILITY / "cluster_bootstrap.py"
    spec = importlib.util.spec_from_file_location("query_multibackbone_cluster_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = load_bootstrap_module()
METRICS = ("auroc", "auprc", "average_precision")
DISPLAY = dict(zip(TASKS, ("CSLV", "CDLV", "CCOM")))
PRIMARY_METRICS = METRICS[:2]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_arrays(loaded):
    return dict(image_paths=np.asarray([r["image_path"] for r in loaded["test"]]),
                patient_ids=np.asarray([r["patient_id"] for r in loaded["test"]]),
                labels=np.asarray([[r[task] for task in TASKS] for r in loaded["test"]], dtype=np.int8),
                task_keys=np.asarray(TASKS), class_names=np.asarray(loaded["classes"]))


def verify_prediction(data, expected, branches, expected_seed=None):
    identity_keys = ["image_paths", "patient_ids", "labels"]
    identity_keys.extend(key for key in ("task_keys", "class_names") if key in expected)
    for key in identity_keys:
        if key not in data or not np.array_equal(data[key], expected[key]):
            raise ValueError("Prediction/frozen test identity mismatch: " + key)
    if "task_keys" in data and not np.array_equal(data["task_keys"], np.asarray(TASKS)):
        raise ValueError("Prediction task_keys order differs")
    if "task_names" in data and not np.array_equal(data["task_names"], np.asarray(TASKS)):
        raise ValueError("Prediction task order differs")
    if expected_seed is not None:
        seed = np.asarray(data.get("seed"))
        if seed.shape != () or seed.dtype.kind not in "iu" or int(seed) != expected_seed:
            raise ValueError("Prediction support seed differs")
    unexpected = {"readout_only", "fixed_aggregation"}.intersection(data).difference(branches)
    if unexpected:
        raise ValueError("Prediction archive contains the wrong control: " + repr(unexpected))
    for branch in branches:
        value = data.get(branch)
        if value is None or value.shape != expected["labels"].shape or not np.isfinite(value).all():
            raise ValueError("Invalid branch scores: " + branch)


def read_model(run, model, control, expected, score_domain=None):
    branches = ("original", control, "query")
    data, provenance = {}, {}
    for seed in SEEDS:
        path = Path(run) / model / f"seed_{seed}" / "predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            current = {key: archive[key] for key in archive.files}
        verify_prediction(current, expected, branches, expected_seed=seed)
        if score_domain and "probability" in score_domain:
            for branch in branches:
                if np.any(current[branch] < 0) or np.any(current[branch] > 1):
                    raise ValueError(f"{model}: declared probabilities outside [0, 1]")
        if model == "RadZero":
            logit_path = path.with_name("logits.npz")
            with np.load(logit_path, allow_pickle=False) as archive:
                for branch in branches:
                    if not np.array_equal(current[branch], archive[branch]):
                        raise ValueError("RadZero predictions must retain native raw logits")
            provenance[str(logit_path.resolve())] = sha(logit_path)
        seed_result_path = path.with_name("result.json")
        if seed_result_path.exists():
            record = json.loads(seed_result_path.read_text())
            if (record.get("status") != "complete" or record.get("model") != model or
                    record.get("seed") != seed or record.get("rows") != len(expected["labels"]) or
                    record.get("prediction_sha256") != sha(path)):
                raise ValueError("Per-seed completion/provenance mismatch: " + str(seed_result_path))
            provenance[str(seed_result_path.resolve())] = sha(seed_result_path)
        if data and not np.array_equal(current["original"], data[SEEDS[0]]["original"]):
            raise ValueError(f"{model}: Original changed across support seeds")
        data[seed] = current
        provenance[str(path.resolve())] = sha(path)
    return data, provenance


def validate_run_plan(path, models, protocol_path, experiment):
    """Require the exact declared model/control scope, including negative runs."""
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if plan.get("models") != list(models):
        raise ValueError("Statistics model scope/order differs from frozen run plan")
    if (Path(plan.get("protocol", "")).resolve() != Path(protocol_path).resolve() or
            plan.get("protocol_sha256") != sha(protocol_path)):
        raise ValueError("Run plan protocol identity differs")
    for key, expected in (("support_seeds", list(SEEDS)), ("tasks", list(TASKS)),
                          ("eta", .3), ("views", ["identity"]),
                          ("all_models_tasks_seeds_retained", True), ("statistics_after_all_jobs", True)):
        if plan.get(key) != expected:
            raise ValueError("Run plan changed frozen scope: " + key)
    expected_controls = {model: ("Q - R (readout only)" if experiment["model_controls"][model]["control_name"] == "readout_only"
                                else "Q - F (fixed visual attention outputs)") for model in models}
    if plan.get("primary_contrast") != expected_controls:
        raise ValueError("Run plan conflates or changes the model-specific controls")
    return dict(path=str(path), sha256=sha(path), verified_models=list(models),
                verified_primary_contrast=expected_controls)


def validate_worker_metadata(run, model, loaded, protocol_path):
    """Check the real worker schemas and native score domains before scoring."""
    folder = Path(run) / model
    paths = {name: folder / name for name in ("result.json", "input_audit.json", "native_parity.json", "source_sha256.json")}
    records = {name: json.loads(path.read_text()) for name, path in paths.items()}
    result, identity = records["result.json"], records["input_audit.json"]
    if (result.get("status") != "complete" or result.get("model") != model or
            result.get("rows") != len(loaded["test"]) or (folder / "failure.json").exists()):
        raise ValueError("Worker has not completed the full declared test: " + model)
    if result.get("seeds", result.get("support_seeds")) != list(SEEDS):
        raise ValueError("Worker support seed scope differs: " + model)
    if (identity.get("protocol_sha256") != sha(protocol_path) or
            Path(identity.get("protocol", "")).resolve() != Path(protocol_path).resolve() or
            identity.get("task_keys") != list(TASKS) or identity.get("class_names") != loaded["classes"] or
            identity.get("seeds", identity.get("support_seeds")) != list(SEEDS) or
            identity.get("support_test_patient_overlap") != 0 or
            identity.get("inference_images", identity.get("test_images")) != len(loaded["test"])):
        raise ValueError("Worker input audit mismatches frozen identities: " + model)
    if result.get("native_parity") != records["native_parity.json"]:
        raise ValueError("Native parity report differs from completion record: " + model)
    # Assert hashes of the protocol and inference sources saved by each worker.
    for source, expected in records["source_sha256.json"].items():
        if sha(source) != expected:
            raise ValueError("Worker source/protocol changed: " + source)
    control = loaded["master"]["experiment"]["model_controls"][model]["control_name"]
    metadata = result.get("metadata", {})
    if model == "AlphaRAD":
        if (control != "readout_only" or metadata.get("control") != control or
                metadata.get("classification_aggregation") != "single native logit then sigmoid" or
                metadata.get("primary_contrast") != "query - readout_only"):
            raise ValueError("AlphaRAD native scoring/control metadata differs")
        score_domain = "native_full_CPU_float32_sigmoid_probability"
    elif model == "RadZero":
        if (control != "readout_only" or metadata.get("score_domain") != "native_raw_logits" or
                metadata.get("primary_comparison") != "query_minus_readout_only" or
                metadata.get("arms") != ["original", "readout_only", "query"] or metadata.get("views") != ["identity"]):
            raise ValueError("RadZero native scoring/control metadata differs")
        score_domain = "native_raw_logits"
    else:
        if (control != "fixed_aggregation" or result.get("control_name") != control or
                result.get("comparison") != "Q-F" or result.get("readout_only_R_supported") is not False or
                result.get("task_keys") != list(TASKS) or result.get("class_names") != loaded["classes"] or
                result.get("views") != ["identity"]):
            raise ValueError("Decoder worker control/task metadata differs: " + model)
        paths["metadata.json"] = folder / "metadata.json"
        baseline = json.loads(paths["metadata.json"].read_text())
        score_domain = baseline["native"]["score_kind"]
        if ((model == "CARZero" and "logit" not in score_domain) or
                (model != "CARZero" and "probability" not in score_domain)):
            raise ValueError("Decoder native score domain differs: " + model)
        for seed in SEEDS:
            for task in TASKS:
                item = baseline["interventions"][str(seed)][task]
                if item.get("control_name") != control or item.get("contrast") != "Q-F" or item.get("readout_only_R_supported") is not False:
                    raise ValueError("Per-task decoder control definition differs")
    # Match each seed's actual fitter inputs, not only summary counts.
    for seed in SEEDS:
        path = folder / f"seed_{seed}" / "support_manifest.json"
        support = json.loads(path.read_text())
        if (support.get("seed") != seed or support.get("task_support_rows") != loaded["supports"][seed] or
                support.get("eta") != .3 or support.get("test_labels_used_for_fit") is not False):
            raise ValueError("Worker fitted support identity differs: " + str(path))
        paths[f"seed_{seed}/support_manifest.json"] = path
    return dict(status="verified", control_name=control, native_score_domain=score_domain,
                completion_and_metadata_sha256={str(path.resolve()): sha(path) for path in paths.values()})


def sklearn_check(labels, scores, points):
    error = 0.0
    for b, value in enumerate(scores.values()):
        for task in range(labels.shape[1]):
            y, score = labels[:, task], value[:, task]
            precision, recall, _ = precision_recall_curve(y, score)
            independent = np.asarray([roc_auc_score(y, score), auc(recall, precision), average_precision_score(y, score)])
            error = max(error, float(np.abs(points[b, task] - independent).max()))
    if error > 1e-12:
        raise ValueError(f"Independent sklearn point-metric audit failed: {error}")
    return error


def model_bootstrap(data, control, expected, plan, output, provenance):
    """Bootstrap Original once, retaining the same draws for all seed branches."""
    scores = {"original": data[SEEDS[0]]["original"]}
    for seed in SEEDS:
        for branch in (control, "query"):
            scores[f"seed_{seed}__{branch}"] = data[seed][branch]
    fingerprint = dict(predictions_sha256=provenance, draw_plan_sha256=plan.sha256,
                       summarize_source_sha256=sha(__file__), bootstrap_source_sha256=sha(STABILITY / "cluster_bootstrap.py"),
                       branch_order=list(scores), task_order=list(TASKS))
    fingerprint_path = output / "INPUT_FINGERPRINT.json"
    metrics_path = output / "bootstrap_metrics.npz"
    used_cache = False
    if fingerprint_path.exists() and metrics_path.exists():
        saved = json.loads(fingerprint_path.read_text())
        if saved.get("inputs") == fingerprint and saved.get("metrics_sha256") == sha(metrics_path):
            with np.load(metrics_path, allow_pickle=False) as archive:
                points = archive["point_metrics"]
                draws = archive["replicate_metrics"]
                if not np.array_equal(archive["branch_names"], np.asarray(list(scores))):
                    raise ValueError("Cached bootstrap branch order mismatch")
            used_cache = True
    if not used_cache:
        result = bootstrap.paired_cluster_bootstrap(expected["labels"], scores, expected["patient_ids"],
                                                   task_names=TASKS, plan=plan)
        points, draws = result.point_metrics, result.replicate_metrics
        result.summary["input_provenance"] = fingerprint
        bootstrap.save_bootstrap(result, output)
        write_json(fingerprint_path, dict(inputs=fingerprint, metrics_sha256=sha(metrics_path)))
    error = sklearn_check(expected["labels"], scores, points)
    # [seed, branch, task, metric] and [seed, replicate, branch, task, metric].
    seed_points, seed_draws = [], []
    for i, seed in enumerate(SEEDS):
        indices = [0, 1 + 2 * i, 2 + 2 * i]
        seed_points.append(points[indices])
        seed_draws.append(draws[:, indices])
    return np.asarray(seed_points), np.asarray(seed_draws), dict(
        cached=used_cache, sklearn_point_metric_max_abs_error=error, input_provenance=fingerprint)


def scoped_metrics(points, draws):
    """Append a task macro to each seed/replicate; never ignore missing tasks."""
    # points [S,B,T,M], draws [S,R,B,T,M]
    return (np.concatenate([points, points.mean(axis=2, keepdims=True)], axis=2),
            np.concatenate([draws, draws.mean(axis=3, keepdims=True)], axis=3))


def ci_record(point, samples, scale=100.0):
    value = bootstrap.interval_summary(point, samples)
    return dict(point=None if value["point"] is None else scale * value["point"],
                ci_low=None if value["ci"] is None else scale * value["ci"][0],
                ci_high=None if value["ci"] is None else scale * value["ci"][1],
                valid_replicates=value["valid_replicates"], invalid_replicates=value["invalid_replicates"],
                valid_fraction=value["valid_fraction"], confidence_level=value["confidence_level"])


def summarize_model(model, control_definition, points, draws):
    control = control_definition["control_name"]
    branches = ("original", control, "query")
    scopes = tuple(TASKS) + ("macro_three_tasks",)
    points, draws = scoped_metrics(points, draws)
    means, sds = points.mean(axis=0), points.std(axis=0, ddof=1)
    mean_draws = draws.mean(axis=0)
    mean_rows, paired_rows, per_seed, per_seed_pairs = [], [], [], []
    comparisons = ((2, 1, control_definition["contrast_type"]),
                   (2, 0, "query_minus_original"), (1, 0, control + "_minus_original"))
    base = dict(model=model, control_name=control, contrast_type=control_definition["contrast_type"])
    for t, task in enumerate(scopes):
        for b, branch in enumerate(branches):
            for k, metric in enumerate(METRICS):
                ci = ci_record(means[b, t, k], mean_draws[:, b, t, k])
                mean_rows.append(dict(**base, task=task, branch=branch, metric=metric,
                    mean_percent=ci.pop("point"), support_seed_sd_pp=float(100 * sds[b, t, k]),
                    ci_low_percent=ci.pop("ci_low"), ci_high_percent=ci.pop("ci_high"), **ci))
                for s, seed in enumerate(SEEDS):
                    per_seed.append(dict(**base, support_seed=seed, task=task, branch=branch,
                                         metric=metric, metric_percent=float(100 * points[s, b, t, k])))
        for hi, lo, comparison in comparisons:
            for k, metric in enumerate(METRICS):
                seed_delta = points[:, hi, t, k] - points[:, lo, t, k]
                sample_delta = mean_draws[:, hi, t, k] - mean_draws[:, lo, t, k]
                ci = ci_record(seed_delta.mean(), sample_delta)
                paired_rows.append(dict(**base, task=task, comparison=comparison, metric=metric,
                    mean_delta_pp=ci.pop("point"), support_seed_delta_sd_pp=float(100 * seed_delta.std(ddof=1)),
                    positive_gain_seed_count=int((seed_delta > 0).sum()), negative_gain_seed_count=int((seed_delta < 0).sum()),
                    minimum_seed_delta_pp=float(100 * seed_delta.min()), maximum_seed_delta_pp=float(100 * seed_delta.max()),
                    ci_low_pp=ci.pop("ci_low"), ci_high_pp=ci.pop("ci_high"), **ci))
                for s, seed in enumerate(SEEDS):
                    seed_ci = ci_record(seed_delta[s], draws[s, :, hi, t, k] - draws[s, :, lo, t, k])
                    per_seed_pairs.append(dict(**base, support_seed=seed, task=task, comparison=comparison,
                        metric=metric, delta_pp=seed_ci.pop("point"), ci_low_pp=seed_ci.pop("ci_low"),
                        ci_high_pp=seed_ci.pop("ci_high"), **seed_ci))
    return mean_rows, paired_rows, per_seed, per_seed_pairs


def text_report(summary, mean_rows, paired_rows):
    text = ["CheXchoNet query-path extension — complete exploratory comparison", "",
            f"Models: {', '.join(summary['models'])}",
            "Fixed test: 7,407 images / 2,470 patients; support seeds: 42–46; each task uses its own 5 positive + 5 negative supports.",
            "Original/R/Q are reported for AlphaRAD and RadZero; Original/F/Q for decoder backbones. Q−F is not Q−R.",
            "All tested model/task/seed results are retained; any model highlighted after seeing these results is exploratory selection.",
            "AUPRC is trapezoidal PR area. AP is an auxiliary separately named metric in CSV/JSON.",
            "Values are percentages; differences/SD are percentage points. CI conditions on the five fixed supports; no multiplicity correction.",
            "Macro averages the three task metrics per seed and per bootstrap draw; no prediction ensemble or cross-model mean.", ""]
    mean_lookup = {(r["model"], r["task"], r["branch"], r["metric"]): r for r in mean_rows}
    pair_lookup = {(r["model"], r["task"], r["comparison"], r["metric"]): r for r in paired_rows}
    for model in summary["models"]:
        definition = MODEL_CONTROLS[model]
        text.append(model + " — " + definition["contrast_type"])
        for task in (*TASKS, "macro_three_tasks"):
            text.append("  " + DISPLAY.get(task, task))
            for branch in ("original", definition["control_name"], "query"):
                cells = []
                for metric in PRIMARY_METRICS:
                    r = mean_lookup[model, task, branch, metric]
                    cells.append(f"{metric.upper()} {r['mean_percent']:.4f} ± {r['support_seed_sd_pp']:.4f}")
                text.append("    " + branch + ": " + "; ".join(cells))
            for contrast in (definition["contrast_type"], "query_minus_original", definition["control_name"] + "_minus_original"):
                cells = []
                for metric in PRIMARY_METRICS:
                    r = pair_lookup[model, task, contrast, metric]
                    ci = ("unavailable" if r["ci_low_pp"] is None else f"[{r['ci_low_pp']:+.4f}, {r['ci_high_pp']:+.4f}]")
                    cells.append(f"{metric.upper()} {r['mean_delta_pp']:+.4f} {ci}; positive {r['positive_gain_seed_count']}/5")
                text.append("    " + contrast + ": " + "; ".join(cells))
        text.append("")
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol_v1/catalog.json")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--run-plan", type=Path, help="Frozen plan; defaults to RUN/FROZEN_RUN_PLAN.json")
    args = parser.parse_args()
    run = args.run.resolve()
    verify_frozen(args.protocol)
    loaded = inherited_loader().load_protocol(args.protocol)
    experiment = loaded["master"]["experiment"]
    models = args.models
    if not models or len(models) != len(set(models)) or not set(models) <= set(experiment["model_controls"]):
        raise ValueError("Model list differs from the frozen eligible definitions")
    if args.replicates != experiment["bootstrap"]["replicates"] or args.seed != experiment["bootstrap"]["seed"]:
        raise ValueError("Bootstrap settings differ from frozen protocol")
    run_plan = args.run_plan or run / "FROZEN_RUN_PLAN.json"
    plan_audit = validate_run_plan(run_plan, models, args.protocol, experiment)
    # Require all declared model/seed archives before reading any metrics. No
    # successful-only model selection is performed by this summary program.
    missing = [str(run / model / f"seed_{seed}" / "predictions.npz") for model in models for seed in SEEDS
               if not (run / model / f"seed_{seed}" / "predictions.npz").is_file()]
    if missing:
        raise FileNotFoundError("Full model/seed predictions required: " + ", ".join(missing))
    # Validate every worker's complete metadata before any model is scored.
    worker_audits = {model: validate_worker_metadata(run, model, loaded, args.protocol) for model in models}
    expected = expected_arrays(loaded)
    plan = bootstrap.make_patient_draw_plan(expected["patient_ids"], n_resamples=args.replicates, seed=args.seed)
    bootstrap.save_patient_draw_plan(plan, run / "patient_draws.npz")
    summary = dict(status="complete", created_at_utc=datetime.now(timezone.utc).isoformat(),
        study_type=experiment["study_type"], protocol=str(args.protocol.resolve()), protocol_sha256=sha(args.protocol),
        models=models, model_controls={model: experiment["model_controls"][model] for model in models},
        task_order=list(TASKS), support_seeds=list(SEEDS), test_images=7407, test_patients=2470,
        bootstrap_replicates=args.replicates, bootstrap_seed=args.seed, draw_plan_sha256=plan.sha256,
        metric_names=list(METRICS), primary_metrics=list(PRIMARY_METRICS),
        uncertainty_scope=experiment["uncertainty_scope"], outcome_selection_note=experiment["outcome_selection_note"],
        data_use_history=experiment["data_use_history"], summary_policy=experiment["summary_policy"],
        model_interface_policy=experiment["model_interface_policy"], no_cross_model_averaging=True,
        model_audits={}, source_sha256={str(Path(__file__).resolve()): sha(__file__),
                                     str(STABILITY / "cluster_bootstrap.py"): sha(STABILITY / "cluster_bootstrap.py")})
    summary["run_plan"] = plan_audit["path"]
    summary["run_plan_sha256"] = plan_audit["sha256"]
    summary["run_plan_audit"] = plan_audit
    summary["worker_metadata_audits"] = worker_audits
    result_groups = [[], [], [], []]
    for model in models:
        print("BOOTSTRAP", model, flush=True)
        definition = experiment["model_controls"][model]
        data, provenance = read_model(run, model, definition["control_name"], expected,
                                      score_domain=worker_audits[model]["native_score_domain"])
        points, draws, audit = model_bootstrap(data, definition["control_name"], expected, plan,
                                              run / "bootstrap" / model, provenance)
        summary["model_audits"][model] = audit
        for accumulated, values in zip(result_groups, summarize_model(model, definition, points, draws)):
            accumulated.extend(values)
    mean_rows, paired_rows, per_seed, per_seed_pairs = result_groups
    for filename, rows in zip(("MEAN_SD_COMPARISON.csv", "PAIRED_COMPARISONS.csv", "PER_SEED_RESULTS.csv", "PER_SEED_PAIRED_COMPARISONS.csv"), result_groups):
        write_csv(run / filename, rows)
    summary.update(mean_sd_results=mean_rows, paired_comparisons=paired_rows,
                   per_seed_results=per_seed, per_seed_paired_comparisons=per_seed_pairs)
    write_json(run / "RESULTS.json", summary)
    (run / "RESULTS.txt").write_text(text_report(summary, mean_rows, paired_rows))
    print(json.dumps(dict(status="complete", run=str(run), models=models,
                          method_means=len(mean_rows), paired_comparisons=len(paired_rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
