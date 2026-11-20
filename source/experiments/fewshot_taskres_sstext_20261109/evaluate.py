#!/usr/bin/env python3
"""Score only after every predeclared unit has an authenticated prediction commit.

The paired bootstrap conditions on the five existing support draws. One group
draw plan per dataset is reused across models, methods and support draws; metrics
are averaged over support draws, never computed from averaged predictions.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

try:
    from . import data_interface as data
except ImportError:
    import data_interface as data

ENGINE_PATH = data.ROOT / "experiments/chexchonet_stability_20261024/cluster_bootstrap.py"
_spec = importlib.util.spec_from_file_location("_taskres_paired_bootstrap", ENGINE_PATH)
engine = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = engine
_spec.loader.exec_module(engine)

MODELS, DATASETS, SEEDS = data.MODELS, data.DATASETS, data.SEEDS
METHODS = ("original", "query", "view", "query_view", "feature_text_zero_shot", "taskres", "sstext_plus")
DISPLAY = dict(original="Original", query="Q", view="V", query_view="Q+V",
               feature_text_zero_shot="Feature-text zero-shot", taskres="TaskRes feature OvR",
               sstext_plus="SS-Text+ feature OvR")
COMPARISONS = (("taskres", "original"), ("sstext_plus", "original"),
               ("taskres", "feature_text_zero_shot"), ("sstext_plus", "feature_text_zero_shot"),
               ("query", "original"), ("query", "taskres"), ("query", "sstext_plus"),
               ("query_view", "taskres"), ("query_view", "sstext_plus"))
UNCERTAINTY_NOTE = (
    "Means average five separately fitted support-draw metrics; predictions are never averaged or concatenated. "
    "Sample SD (ddof=1) describes those five support draws. Paired 95% percentile intervals quantify test-group "
    "sampling conditional on the five fixed support sets, not uncertainty over new support samples. "
    "All predeclared comparisons are retained without multiple-comparison correction or best-seed selection. "
    "These previously exposed datasets do not constitute independent external validation.")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checked(path, expected):
    path = Path(path)
    if data.sha256(path) != expected:
        raise ValueError(f"Committed source/artifact SHA256 mismatch: {path}")
    return path


def verify_run_inputs(run_path, bootstrap_replicates=2000, bootstrap_seed=20261024):
    """Validate *all* requested prediction commits before any annotation read."""
    run = Path(run_path).absolute()
    expected_ids = tuple(f"{model}_{dataset}/seed_{seed}" for model in MODELS
                         for dataset in DATASETS for seed in SEEDS)
    barrier_path, snapshot_path = run / "predictions_complete.json", run / "experiment_snapshot.json"
    if not barrier_path.is_file():
        raise ValueError("All-unit prediction commit barrier is missing; test labels remain sealed")
    snapshot_digest = data.sha256(snapshot_path)
    barrier, snapshot = read_json(barrier_path), read_json(snapshot_path)
    if (barrier.get("status") != "all_predictions_committed"
            or set(barrier.get("units", {})) != set(expected_ids)
            or barrier.get("run_snapshot_sha256") != snapshot_digest):
        raise ValueError("Incomplete/mismatched all-unit prediction commit barrier")
    config = snapshot.get("config")
    if not isinstance(config, dict):
        raise ValueError("Run snapshot lacks its frozen evaluation configuration")
    declared_scope = dict(methods=list(METHODS), models=list(MODELS), datasets=list(DATASETS),
                          seeds=list(SEEDS), comparisons=[list(pair) for pair in COMPARISONS])
    for key, expected in declared_scope.items():
        if config.get(key) != expected:
            raise ValueError("Frozen evaluation configuration mismatch: " + key)
    bootstrap = config.get("bootstrap", {})
    if bootstrap.get("replicates") != bootstrap_replicates or bootstrap.get("seed") != bootstrap_seed:
        raise ValueError("Frozen bootstrap replicates/seed differ from requested evaluation")
    if set(bootstrap.get("unit_by_dataset", {})) != set(DATASETS):
        raise ValueError("Frozen bootstrap grouping must cover exactly the declared datasets")
    snapshot_sources = snapshot.get("source_sha256")
    if not isinstance(snapshot_sources, dict) or not snapshot_sources:
        raise ValueError("Run snapshot lacks frozen source locks")
    snapshot_units = snapshot.get("unit_manifests", {})
    if set(snapshot_units) != set(expected_ids):
        raise ValueError("Run snapshot lacks the exact frozen unit manifests")
    for filename, digest in snapshot_sources.items():
        checked(filename, digest)
    catalog_path = config.get("catalog_path", data.DEFAULT_CATALOG)
    source_run = config.get("source_run", data.DEFAULT_SOURCE_RUN)
    loaded = {}
    dataset_references = {}
    for model in MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                unit_id = f"{model}_{dataset}/seed_{seed}"
                folder = run / unit_id
                commit_path = checked(folder / "fit_commit.json", barrier["units"][unit_id]["fit_commit_sha256"])
                commit = read_json(commit_path)
                if (commit.get("unit_id") != unit_id or commit.get("run_snapshot_sha256") != snapshot_digest
                        or commit.get("test_labels_opened") is not False):
                    raise ValueError("Wrong unit/run or labels opened before prediction commit: " + unit_id)
                if commit.get("source_sha256") != snapshot_sources:
                    raise ValueError("Prediction commit source locks differ from frozen snapshot: " + unit_id)
                artifacts = commit.get("artifacts", {})
                if not {"fit_info.json", "predictions.npz", "taskres_state.npz", "sstext_state.npz"}.issubset(artifacts):
                    raise ValueError("Prediction commit omits required fit artifacts: " + unit_id)
                for filename, digest in artifacts.items():
                    if Path(filename).name != filename:
                        raise ValueError("Unsafe fit artifact path")
                    checked(folder / filename, digest)
                unit = data.load_unit(model, dataset, seed, catalog_path, source_run)
                if (unit.manifest() != snapshot_units[unit_id]
                        or commit.get("unit_manifest_sha256") != data.object_sha256(snapshot_units[unit_id])):
                    raise ValueError("Prediction unit provenance differs from authenticated cache: " + unit_id)
                if bootstrap["unit_by_dataset"][dataset] != unit.provenance["resampling_unit"]:
                    raise ValueError("Frozen bootstrap grouping differs from authenticated dataset protocol")
                with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
                    allowed = {"image_paths", "class_names", "group_ids", *METHODS}
                    if set(arrays.files) != allowed:
                        raise ValueError("Prediction archive has missing/unexpected fields or test labels")
                    predictions = {key: arrays[key] for key in arrays.files}
                for key, values in (("image_paths", unit.test_paths), ("class_names", unit.classes),
                                    ("group_ids", unit.test_groups)):
                    if not np.array_equal(predictions[key], np.asarray(values)):
                        raise ValueError("Prediction identity/order mismatch: " + key)
                for method in METHODS:
                    scores = predictions[method]
                    if scores.shape != (len(unit.test_paths), len(unit.classes)) or not np.isfinite(scores).all():
                        raise ValueError("Invalid prediction shape/nonfinite values: " + method)
                for method in data.NATIVE_METHODS:
                    if not np.array_equal(predictions[method], unit.native_predictions[method]):
                        raise ValueError("Native control differs from same-support source cache: " + method)
                reference = dataset_references.setdefault(dataset, unit)
                if (unit.test_paths != reference.test_paths or unit.test_groups != reference.test_groups
                        or unit.classes != reference.classes or unit._test_manifest != reference._test_manifest):
                    raise ValueError("Dataset cohort/classes/label manifest differ across model/support draws")
                loaded[unit_id] = (unit, predictions, commit)
    return loaded, dict(run_snapshot_sha256=snapshot_digest,
                        predictions_complete_sha256=data.sha256(barrier_path),
                        source_sha256={str(Path(__file__).resolve()): data.sha256(__file__),
                                       str(ENGINE_PATH): data.sha256(ENGINE_PATH),
                                       str(Path(data.__file__).resolve()): data.sha256(data.__file__)})


def interval_columns(prefix, point, replicates):
    result = engine.interval_summary(float(point), np.asarray(replicates, dtype=np.float64))
    ci = result["ci"]
    return {prefix + "_ci_low": None if ci is None else 100 * ci[0],
            prefix + "_ci_high": None if ci is None else 100 * ci[1],
            prefix + "_valid_replicates": result["valid_replicates"],
            prefix + "_invalid_replicates": result["invalid_replicates"]}


def summarize_arrays(model, dataset, points, replicates, class_names, resampling_unit,
                     *, methods=METHODS, seeds=SEEDS, comparisons=COMPARISONS):
    """points[S,M,C,K], replicates[S,R,M,C,K]; preserve paired replicate axis."""
    if points.shape != (len(seeds), len(methods), len(class_names), len(engine.METRIC_NAMES)):
        raise ValueError("Unexpected point-metric axes")
    if replicates.ndim != 5 or replicates.shape[0] != len(seeds) or replicates.shape[2:] != points.shape[1:]:
        raise ValueError("Unexpected bootstrap-metric axes")
    if len(seeds) < 2 or not np.isfinite(points).all():
        raise ValueError("Need at least two complete support-draw metrics")
    # A replicate with an invalid class is invalid for the fixed-class macro.
    # A missing seed likewise invalidates the five-seed mean; never nanmean.
    macro_points, macro_replicates = points.mean(axis=2), replicates.mean(axis=3)
    means, per_seed, pairs, per_class = [], [], [], []
    base = dict(model=model, dataset=dataset, resampling_unit=resampling_unit)
    for imethod, method in enumerate(methods):
        row = dict(base, method=method, seed_count=len(seeds), views=2 if method in ("view", "query_view") else 1)
        for imetric, metric in enumerate(engine.METRIC_NAMES):
            values = macro_points[:, imethod, imetric]
            row.update({metric + "_mean_percent": 100 * float(values.mean()),
                        metric + "_support_sd_pp": 100 * float(values.std(ddof=1)),
                        **interval_columns(metric + "_percent", values.mean(),
                                           macro_replicates[:, :, imethod, imetric].mean(axis=0))})
        means.append(row)
        for iseed, seed in enumerate(seeds):
            per_seed.append(dict(base, method=method, seed=seed, **{
                metric + "_percent": 100 * float(macro_points[iseed, imethod, k])
                for k, metric in enumerate(engine.METRIC_NAMES)}))
        for iclass, class_name in enumerate(class_names):
            row = dict(base, method=method, class_name=class_name)
            for imetric, metric in enumerate(engine.METRIC_NAMES):
                values = points[:, imethod, iclass, imetric]
                row.update({metric + "_mean_percent": 100 * float(values.mean()),
                            metric + "_support_sd_pp": 100 * float(values.std(ddof=1)),
                            **interval_columns(metric + "_percent", values.mean(),
                                               replicates[:, :, imethod, iclass, imetric].mean(axis=0))})
            per_class.append(row)
    for target, reference in comparisons:
        ti, ri = methods.index(target), methods.index(reference)
        gains = macro_points[:, ti] - macro_points[:, ri]
        draws = (macro_replicates[:, :, ti] - macro_replicates[:, :, ri]).mean(axis=0)
        row = dict(base, target=target, reference=reference, seed_count=len(seeds),
                   comparison_scope="two_vs_one_views" if target == "query_view" else "matched_single_view",
                   both_primary_positive_seeds=int(np.all(gains[:, :2] > 0, axis=1).sum()))
        for imetric, metric in enumerate(engine.METRIC_NAMES):
            values = gains[:, imetric]
            row.update({metric + "_delta_pp": 100 * float(values.mean()),
                        metric + "_gain_sd_pp": 100 * float(values.std(ddof=1)),
                        metric + "_positive_seeds": int((values > 0).sum()),
                        metric + "_negative_seeds": int((values < 0).sum()),
                        metric + "_ties": int((values == 0).sum()),
                        **interval_columns(metric + "_delta_pp", values.mean(), draws[:, imetric])})
        pairs.append(row)
    return dict(means=means, per_seed=per_seed, comparisons=pairs, per_class=per_class)


def _sklearn_point_audit(labels, predictions, point_metrics):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
    largest = 0.0
    for imethod, method in enumerate(METHODS):
        for iclass in range(labels.shape[1]):
            y, score = labels[:, iclass], predictions[method][:, iclass]
            precision, recall, _ = precision_recall_curve(y, score)
            reference = np.asarray([roc_auc_score(y, score), auc(recall, precision), average_precision_score(y, score)])
            largest = max(largest, float(np.max(np.abs(reference - point_metrics[imethod, iclass]))))
    if largest > 1e-12:
        raise ValueError("Independent sklearn metric audit failed")
    return largest


def _ci_text(row, metric):
    values = [row[metric + "_delta_pp"], row[metric + "_delta_pp_ci_low"], row[metric + "_delta_pp_ci_high"]]
    formatted = ["NA" if v is None else f"{v:+.2f}" for v in values]
    return f"{formatted[0]} [{formatted[1]}, {formatted[2]}]"


def _report(summary):
    lines = ["TaskRes / SS-Text+ supplemental comparison", "", UNCERTAINTY_NOTE, "",
             "Metrics: class-macro AUROC and trapezoidal AUPRC; average precision is a separate auxiliary metric.",
             "TaskRes and SS-Text+ are declared feature-level one-vs-rest ports; AlphaRAD native controls retain FLaS.",
             "RSNA resamples verified NIH patients; ChestXDet10 resamples image/exact-duplicate groups, not verified patients.",
             "Q is the matched single-view query-adaptation comparison. Q+V uses two views and is separately marked.", "",
             "| Model | Dataset | Method | AUROC mean ± support SD | AUPRC mean ± support SD |",
             "|---|---|---|---:|---:|"]
    for row in summary["means"]:
        lines.append("| " + " | ".join([row["model"], row["dataset"], DISPLAY[row["method"]],
            f"{row['auroc_mean_percent']:.2f} ± {row['auroc_support_sd_pp']:.2f}",
            f"{row['auprc_mean_percent']:.2f} ± {row['auprc_support_sd_pp']:.2f}"]) + " |")
    lines.extend(["", "| Model | Dataset | Target − reference | View budget | ΔAUROC [95% CI] | ΔAUPRC [95% CI] |",
                  "|---|---|---|---|---:|---:|"])
    for row in summary["comparisons"]:
        lines.append("| " + " | ".join([row["model"], row["dataset"],
            DISPLAY[row["target"]] + " − " + DISPLAY[row["reference"]], row["comparison_scope"],
            _ci_text(row, "auroc"), _ci_text(row, "auprc")]) + " |")
    return "\n".join(lines) + "\n"


def evaluate_run(run_path, bootstrap_replicates=2000, bootstrap_seed=20261024):
    run = Path(run_path).absolute()
    loaded, provenance = verify_run_inputs(run, bootstrap_replicates, bootstrap_seed)
    signature = dict(provenance, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
                     methods=list(METHODS), comparisons=[list(pair) for pair in COMPARISONS])
    final_path = run / "RESULTS.json"
    if final_path.exists():
        previous = read_json(final_path)
        if previous.get("evaluation_signature") != signature:
            raise ValueError("Existing results have different provenance/bootstrap settings; use a new run")
        for filename, digest in previous.get("output_sha256", {}).items():
            checked(run / filename, digest)
        return previous
    # This is the only annotation-unlock boundary. verify_run_inputs above has
    # validated all units, all fits, all source bytes and every native prediction.
    labels_by_dataset, plans, plan_paths = {}, {}, {}
    for dataset in DATASETS:
        unit = loaded[f"{MODELS[0]}_{dataset}/seed_{SEEDS[0]}"][0]
        labels_by_dataset[dataset] = data.load_test_labels(unit)
        plan = engine.make_patient_draw_plan(unit.test_groups, n_resamples=bootstrap_replicates, seed=bootstrap_seed)
        plans[dataset] = plan
        path = run / "bootstrap" / f"{dataset}_group_plan.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, group_ids=plan.patient_ids, counts=plan.counts, seed=np.asarray(plan.seed),
                            sha256=np.asarray(plan.sha256), resampling_unit=np.asarray(unit.provenance["resampling_unit"]))
        plan_paths[str(path.relative_to(run))] = data.sha256(path)
    records = dict(means=[], per_seed=[], comparisons=[], per_class=[])
    archives, point_audits = {}, {}
    for model in MODELS:
        for dataset in DATASETS:
            points, draws = [], []
            for seed in SEEDS:
                unit_id = f"{model}_{dataset}/seed_{seed}"
                unit, predictions, commit = loaded[unit_id]
                result = engine.paired_cluster_bootstrap(labels_by_dataset[dataset],
                    {method: predictions[method] for method in METHODS}, unit.test_groups,
                    task_names=unit.classes, plan=plans[dataset])
                point_audits[unit_id] = _sklearn_point_audit(labels_by_dataset[dataset], predictions, result.point_metrics)
                path = run / unit_id / "bootstrap_metrics.npz"
                np.savez_compressed(path, point_metrics=result.point_metrics, replicate_metrics=result.replicate_metrics,
                    method_names=np.asarray(METHODS), class_names=np.asarray(unit.classes),
                    metric_names=np.asarray(engine.METRIC_NAMES), draw_plan_sha256=np.asarray(plans[dataset].sha256),
                    predictions_sha256=np.asarray(commit["artifacts"]["predictions.npz"]))
                archives[str(path.relative_to(run))] = data.sha256(path)
                points.append(result.point_metrics)
                draws.append(result.replicate_metrics)
                print(f"Scored {unit_id}: {bootstrap_replicates} shared paired group draws", flush=True)
            block = summarize_arrays(model, dataset, np.stack(points), np.stack(draws), unit.classes,
                                     unit.provenance["resampling_unit"], methods=METHODS, seeds=SEEDS)
            for key in records:
                records[key].extend(block[key])
    # Refuse to publish a result if commits or source locks changed mid-scoring.
    _, final_provenance = verify_run_inputs(run, bootstrap_replicates, bootstrap_seed)
    if final_provenance != provenance:
        raise ValueError("Run provenance changed during scoring")
    summary = dict(version="taskres-sstext-paired-comparison-v1", status="complete", evaluation_signature=signature,
                   uncertainty_note=UNCERTAINTY_NOTE, metric_units="percent / percentage points",
                   auprc_definition="trapezoidal PR area, not average precision",
                   fixed_seed_count=len(SEEDS), seeds=list(SEEDS), unit_count=len(loaded),
                   bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
                   all_predictions_committed_before_test_labels=True,
                   native_reference_parity="exact cache equality for every unit and native branch",
                   sklearn_point_metric_max_abs_error=point_audits,
                   group_plan_sha256={dataset: plan.sha256 for dataset, plan in plans.items()},
                   **records)
    outputs = {**plan_paths, **archives}
    for key, rows in records.items():
        name = key + ".csv"
        write_csv(run / name, rows)
        outputs[name] = data.sha256(run / name)
    (run / "RESULTS.txt").write_text(_report(summary))
    outputs["RESULTS.txt"] = data.sha256(run / "RESULTS.txt")
    summary["output_sha256"] = outputs
    write_json(final_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261024)
    args = parser.parse_args()
    result = evaluate_run(args.run, args.bootstrap_replicates, args.bootstrap_seed)
    print(json.dumps(dict(status=result["status"], units=result["unit_count"], result=str(args.run / "RESULTS.json"))))


if __name__ == "__main__":
    main()
