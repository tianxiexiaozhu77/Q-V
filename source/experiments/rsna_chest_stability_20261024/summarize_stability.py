#!/usr/bin/env python3
"""Fixed-test support-seed sensitivity and paired cluster confidence intervals.

RSNA clusters are patients. ChestXDet10 has no verified patient IDs: its clusters
are images/exact-duplicate groups, and its intervals are not patient intervals.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
BOOTSTRAP_ROOT = ROOT.parent / "chexchonet_stability_20261024"
sys.path.insert(0, str(BOOTSTRAP_ROOT))
from cluster_bootstrap import (BootstrapResult, METRIC_NAMES, interval_summary,
                               make_patient_draw_plan, paired_cluster_bootstrap,
                               save_bootstrap)

MODELS = ("BiomedCLIP", "AlphaRAD")
DATASETS = ("rsna_pneumonia", "chestxdet10")
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
DISPLAY = dict(zip(BRANCHES, ("原始", "Q", "V", "Q+V")))
OLD_ALIASES = dict(zip(BRANCHES, ("original", "query", "view", "query_view")))
SEEDS = (42, 43, 44, 45, 46)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metric_audit(labels, scores, points):
    """Independent sklearn point estimates, including tied-score PR curves."""
    errors = []
    for ib, branch in enumerate(BRANCHES):
        for ic in range(labels.shape[1]):
            y, p = labels[:, ic], scores[branch][:, ic]
            precision, recall, _ = precision_recall_curve(y, p)
            target = np.asarray([roc_auc_score(y, p), auc(recall, precision),
                                 average_precision_score(y, p)])
            errors.extend(np.abs(points[ib, ic] - target).tolist())
    maximum = float(max(errors))
    if maximum > 1e-12:
        raise ValueError(f"Independent sklearn point-metric discrepancy: {maximum}")
    return maximum


def neutralize_group_language(value):
    """The reused engine names clusters 'patients'; correct Chest metadata."""
    if isinstance(value, dict):
        return {str(k).replace("patients", "groups").replace("patient", "group"):
                neutralize_group_language(v) for k, v in value.items()}
    if isinstance(value, list):
        return [neutralize_group_language(v) for v in value]
    if isinstance(value, str):
        return value.replace("patients", "groups").replace("patient", "group")
    return value


def summarize_uncertainty(point, draws):
    result = interval_summary(float(point), np.asarray(draws))
    return {**result, "point": None if result["point"] is None else result["point"] * 100,
            "ci": None if result["ci"] is None else [x * 100 for x in result["ci"]],
            "unit": "percentage_points"}


def strict_macro(points, draws):
    # Propagate an invalid class instead of silently changing the macro class set.
    return points.mean(axis=-2), draws.mean(axis=-2)


def prepare_expected(spec):
    rows = spec["test_rows"]
    classes = spec["class_names"]
    unit = spec["resampling_unit"]
    if unit not in ("patient", "image_exact_duplicate_group"):
        raise ValueError(f"Unrecognised resampling unit: {unit}")
    expected = {"image_paths": np.asarray([r["image"] for r in rows]),
                "group_ids": np.asarray([str(r["group_id"]) for r in rows]),
                "labels": np.asarray([r["labels"] for r in rows]),
                "class_names": np.asarray(classes)}
    if len(set(expected["image_paths"].tolist())) != len(rows):
        raise ValueError("Duplicate test path")
    if expected["labels"].shape != (len(rows), len(classes)):
        raise ValueError("Unexpected label shape")
    if not np.isin(expected["labels"], (0, 1)).all():
        raise ValueError("Nonbinary test labels")
    if np.any(expected["labels"].sum(axis=0) == 0) or np.any(expected["labels"].sum(axis=0) == len(rows)):
        raise ValueError("Fixed-test class is not evaluable")
    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate class name")
    if unit == "patient":
        patient_to_group, group_to_patient = {}, {}
        for row in rows:
            patient, group = str(row.get("patient_id", "")), str(row["group_id"])
            if not patient or patient == "None":
                raise ValueError("Verified patient ID missing")
            if patient_to_group.setdefault(patient, group) != group or group_to_patient.setdefault(group, patient) != patient:
                raise ValueError("Verified patient IDs and bootstrap groups are not one-to-one")
    manifest = spec.get("test_manifest")
    if isinstance(manifest, dict) and "path" in manifest and "sha256" in manifest:
        if sha(manifest["path"]) != manifest["sha256"]:
            raise ValueError("Fixed test manifest changed")
        frozen_rows = json.loads(Path(manifest["path"]).read_text())["rows"]
        if [r["image"] for r in frozen_rows] != expected["image_paths"].tolist():
            raise ValueError("Protocol rows differ from frozen test identities/order")
        if not np.array_equal(np.asarray([r["labels"] for r in frozen_rows]), expected["labels"]):
            raise ValueError("Protocol rows differ from frozen test labels")
    return expected


def verify_predictions(data, expected):
    for key in ("image_paths", "group_ids", "labels", "class_names"):
        if not np.array_equal(data[key], expected[key]):
            raise ValueError(f"Prediction/protocol mismatch: {key}")
    for branch in BRANCHES:
        values = np.asarray(data[branch])
        if values.shape != expected["labels"].shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores: {branch}")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"Expected native probabilities for {branch}")


def save_group_plan(plan, path, unit):
    # Neutral archive keys are deliberate: an image hash is not a patient ID.
    np.savez_compressed(path, group_ids=plan.patient_ids, counts=plan.counts,
                        seed=np.asarray(plan.seed), sha256=np.asarray(plan.sha256),
                        resampling_unit=np.asarray(unit), generator=np.asarray("numpy.Generator(PCG64)"))


def process(data, expected, spec, plan, folder, provenance):
    verify_predictions(data, expected)
    provenance = {**provenance, "statistics_implementation_sha256": sha(__file__),
                  "bootstrap_engine_sha256": sha(BOOTSTRAP_ROOT / "cluster_bootstrap.py")}
    summary_path = folder / "bootstrap_summary.json"
    archive_path = folder / "bootstrap_metrics.npz"
    if summary_path.exists() and archive_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("provenance") != provenance or summary["draw_plan_sha256"] != plan.sha256:
            raise ValueError("Existing bootstrap cache has changed provenance; choose a new run folder")
        arrays = load_npz(archive_path)
        result = BootstrapResult(summary, arrays["point_metrics"], arrays["replicate_metrics"],
            tuple(arrays["branch_names"]), tuple(arrays["task_names"]), tuple(arrays["metric_names"]), plan)
    else:
        result = paired_cluster_bootstrap(data["labels"], {k: data[k] for k in BRANCHES},
            data["group_ids"], task_names=spec["class_names"], plan=plan)
        if spec["resampling_unit"] != "patient":
            result.summary = neutralize_group_language(result.summary)
        result.summary.update(resampling_unit=spec["resampling_unit"], provenance=provenance,
            patient_identity_verified=spec["resampling_unit"] == "patient",
            uncertainty_scope="test-cluster sampling conditional on this fixed support set and method")
        result.summary["independent_sklearn_point_metric_max_abs_error"] = metric_audit(
            data["labels"], data, result.point_metrics)
        save_bootstrap(result, folder)
    return result


def historical_predictions(model, dataset, expected):
    old_folder = "full_biomed_20261024_v1" if model == "BiomedCLIP" else "full_alpha_20261024_v1"
    path = MEDICAL / "experiments/fewshot_native_query/runs" / old_folder / f"{model}_{dataset}/evaluation/predictions.npz"
    old = load_npz(path)
    if not np.array_equal(old["image_paths"], expected["image_paths"]) or not np.array_equal(old["class_names"], expected["class_names"]):
        raise ValueError("Historical and new fixed tests differ; cannot claim the old fixed-test comparison")
    data = dict(expected)
    data.update({branch: old[alias] for branch, alias in OLD_ALIASES.items()})
    return data, {"predictions_path": str(path), "predictions_sha256": sha(path),
                  "support_protocol": "historical_fixed_support; separate from five new support seeds"}


def per_seed_rows(model, dataset, support_seed, result, unit):
    points, draws = strict_macro(result.point_metrics, result.replicate_metrics)
    output = []
    for ib, branch in enumerate(BRANCHES):
        row = dict(model=model, dataset=dataset, seed=support_seed, branch=branch,
                   resampling_unit=unit)
        for im, metric in enumerate(METRIC_NAMES):
            absolute = summarize_uncertainty(points[ib, im], draws[:, ib, im])
            gain = summarize_uncertainty(points[ib, im] - points[0, im], draws[:, ib, im] - draws[:, 0, im])
            row.update({metric + "_percent": 100 * points[ib, im], metric + "_delta_pp": gain["point"],
                metric + "_ci_low_pp": None if gain["ci"] is None else gain["ci"][0],
                metric + "_ci_high_pp": None if gain["ci"] is None else gain["ci"][1],
                metric + "_valid_fraction": gain["valid_fraction"],
                metric + "_absolute_ci_low_percent": None if absolute["ci"] is None else absolute["ci"][0],
                metric + "_absolute_ci_high_percent": None if absolute["ci"] is None else absolute["ci"][1]})
        output.append(row)
    return output


def aggregate_classification(model, dataset, results, unit):
    points = np.stack([r.point_metrics for r in results])  # seed, branch, class, metric
    draws = np.stack([r.replicate_metrics for r in results])  # seed, replicate, branch, class, metric
    macro_points, macro_draws = strict_macro(points, draws)
    aggregates, rows, per_class = {}, [], []
    for ib, branch in enumerate(BRANCHES):
        row = dict(model=model, dataset=dataset, branch=branch, resampling_unit=unit)
        branch_metrics = {}
        for im, metric in enumerate(METRIC_NAMES):
            values = macro_points[:, ib, im]
            gains = values - macro_points[:, 0, im]
            sample_gains = macro_draws[:, :, ib, im] - macro_draws[:, :, 0, im]
            ci = summarize_uncertainty(gains.mean(), sample_gains.mean(axis=0))
            abs_ci = summarize_uncertainty(values.mean(), macro_draws[:, :, ib, im].mean(axis=0))
            item = dict(mean_percent=float(100 * values.mean()), support_seed_sd_pp=float(100 * values.std(ddof=1)),
                mean_gain_pp=float(100 * gains.mean()), gain_sd_pp=float(100 * gains.std(ddof=1)),
                minimum_gain_pp=float(100 * gains.min()), maximum_gain_pp=float(100 * gains.max()),
                positive_gain_seed_count=int((gains > 0).sum()), mean_gain_paired_cluster_bootstrap=ci,
                mean_metric_cluster_bootstrap=abs_ci)
            branch_metrics[metric] = item
            row.update({metric + "_mean_percent": item["mean_percent"], metric + "_sd_pp": item["support_seed_sd_pp"],
                metric + "_delta_pp": ci["point"], metric + "_ci_low_pp": None if ci["ci"] is None else ci["ci"][0],
                metric + "_ci_high_pp": None if ci["ci"] is None else ci["ci"][1],
                metric + "_positive_seeds": item["positive_gain_seed_count"],
                metric + "_valid_fraction": ci["valid_fraction"]})
        joint_gains = macro_points[:, ib, :2] - macro_points[:, 0, :2]
        row["both_primary_positive_seeds"] = int(np.all(joint_gains > 0, axis=1).sum())
        aggregates[branch] = {"macro": branch_metrics, "both_primary_positive_seeds": row["both_primary_positive_seeds"], "per_class": {}}
        rows.append(row)
        for ic, name in enumerate(results[0].task_names):
            c_row = dict(model=model, dataset=dataset, branch=branch, class_name=name, resampling_unit=unit)
            metrics = {}
            for im, metric in enumerate(METRIC_NAMES):
                values = points[:, ib, ic, im]
                gains = values - points[:, 0, ic, im]
                ci = summarize_uncertainty(gains.mean(), (draws[:, :, ib, ic, im] - draws[:, :, 0, ic, im]).mean(axis=0))
                item = {"mean_percent": float(100 * values.mean()), "support_seed_sd_pp": float(100 * values.std(ddof=1)),
                        "mean_gain_paired_cluster_bootstrap": ci, "positive_gain_seed_count": int((gains > 0).sum())}
                metrics[metric] = item
                c_row.update({metric + "_mean_percent": item["mean_percent"], metric + "_sd_pp": item["support_seed_sd_pp"],
                    metric + "_delta_pp": ci["point"], metric + "_ci_low_pp": None if ci["ci"] is None else ci["ci"][0],
                    metric + "_ci_high_pp": None if ci["ci"] is None else ci["ci"][1], metric + "_valid_fraction": ci["valid_fraction"]})
            aggregates[branch]["per_class"][name] = metrics
            per_class.append(c_row)
    return aggregates, rows, per_class


def spatial_bootstrap(data, expected, plan, unit):
    """Macro PG = class mean of weighted hits/cases; never mean image PG."""
    paths = data["image_paths"].tolist()
    lookup = {x: i for i, x in enumerate(expected["image_paths"].tolist())}
    if len(paths) != len(set(paths)) or any(p not in lookup for p in paths):
        raise ValueError("Spatial test must be a unique subset of the fixed classification test")
    indices = np.asarray([lookup[p] for p in paths])
    if not np.array_equal(data["group_ids"], expected["group_ids"][indices]):
        raise ValueError("Spatial group IDs differ")
    if not np.array_equal(data["class_names"], expected["class_names"]):
        raise ValueError("Spatial class identities differ")
    inverse = np.searchsorted(plan.patient_ids, data["group_ids"])
    if not np.array_equal(plan.patient_ids[inverse], data["group_ids"]):
        raise ValueError("Unknown spatial group")
    weights = plan.counts[:, inverse].astype(np.float64)
    points, draws = [], []
    cases = None
    for branch in BRANCHES:
        hit, case = data[branch + "_hits"], data[branch + "_cases"]
        if hit.shape != (len(paths), len(expected["class_names"])) or case.shape != hit.shape:
            raise ValueError("Invalid spatial hit/case shape")
        if not np.isin(case, (0, 1)).all() or not np.isin(hit, (0, 1)).all() or np.any(hit > case):
            raise ValueError("PG arrays must be binary image/class cases with hit <= case")
        if cases is not None and not np.array_equal(case, cases):
            raise ValueError("Spatial case mask changed between methods")
        cases = case
        if np.any(case.sum(axis=0) == 0):
            raise ValueError("Fixed spatial test has an empty class")
        point = hit.sum(axis=0) / case.sum(axis=0)
        denominators = weights @ case
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled = (weights @ hit) / denominators
        sampled[denominators == 0] = np.nan
        points.append(point)
        draws.append(sampled)
    points, draws = np.stack(points), np.stack(draws, axis=1)
    summary = {"images": len(paths), "groups": len(set(data["group_ids"].tolist())),
        "resampling_unit": unit, "patient_identity_verified": unit == "patient", "n_resamples": len(plan.counts),
        "draw_plan_sha256": plan.sha256, "class_names": data["class_names"].tolist(),
        "cases_per_class": cases.sum(axis=0).astype(int).tolist(),
        "macro_definition": "unweighted mean across fixed classes of sum(image-class hits)/sum(image-class cases)",
        "invalid_draw_policy": "zero-case class is NaN; a macro draw is invalid if any fixed class is invalid",
        "sampling_scope": "reuse classification cohort draw plan; each spatial image receives its group's multiplicity",
        "branches": {}}
    for ib, branch in enumerate(BRANCHES):
        summary["branches"][branch] = {"macro_pg": interval_summary(points[ib].mean(), draws[:, ib].mean(axis=1)),
            "paired_gain": interval_summary(points[ib].mean() - points[0].mean(),
                draws[:, ib].mean(axis=1) - draws[:, 0].mean(axis=1)),
            "per_class": {name: interval_summary(points[ib, ic], draws[:, ib, ic])
                          for ic, name in enumerate(data["class_names"].tolist())}}
    return points, draws, summary


def number(value, signed=False):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def interval_text(row, metric):
    return f"{number(row[metric + '_delta_pp'], True)} [{number(row[metric + '_ci_low_pp'], True)}, {number(row[metric + '_ci_high_pp'], True)}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol_v1/catalog.json")
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.protocol.read_text())
    specs, plans, expected = {}, {}, {}
    source = {str(Path(__file__)): sha(__file__), str(BOOTSTRAP_ROOT / "cluster_bootstrap.py"): sha(BOOTSTRAP_ROOT / "cluster_bootstrap.py"),
              str(args.protocol.resolve()): sha(args.protocol)}
    for dataset in DATASETS:
        entry = catalog["datasets"][dataset]
        path = Path(entry["path"] if isinstance(entry, dict) else entry)
        if isinstance(entry, dict) and "sha256" in entry and sha(path) != entry["sha256"]:
            raise ValueError("Dataset protocol SHA256 changed")
        spec = json.loads(path.read_text())
        source[str(path)] = sha(path)
        if set(spec["seeds"]) != {str(s) for s in SEEDS}:
            raise ValueError("Expected all five predeclared support seeds")
        specs[dataset] = spec
        expected[dataset] = prepare_expected(spec)
        plans[dataset] = make_patient_draw_plan(expected[dataset]["group_ids"], n_resamples=2000, seed=20261024)
        save_group_plan(plans[dataset], args.run / f"{dataset}_cluster_draws.npz", spec["resampling_unit"])
    per_seed, aggregate_rows, class_rows, history_rows, audits = [], [], [], [], []
    spatial_seed_rows, spatial_rows = [], []
    summary = {"status": "running", "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "support_seeds": list(SEEDS), "bootstrap_replicates": 2000, "bootstrap_seed": 20261024,
        "models": list(MODELS), "datasets": {}, "source_sha256": source,
        "uncertainty_note": "Five-seed sample SD measures support sampling sensitivity within the declared candidate pool. "
          "Mean-gain CIs average paired cluster-bootstrap metric differences across the five fixed support sets. "
          "These are conditional test-cluster intervals, not joint support-population/test-population intervals. "
          "No multiplicity adjustment. Every seed retained; no test-driven support or eta selection.",
        "metric_definition": "image-level AUROC and trapezoidal PR AUPRC, class-macro; AP separately reported",
        "historical_support_note": "Historical support is reported separately; it is not automatically identical to the new seed 42."}
    for dataset in DATASETS:
        spec, exp, plan = specs[dataset], expected[dataset], plans[dataset]
        dataset_summary = {"test_images": len(exp["labels"]), "test_groups": len(plan.patient_ids),
            "resampling_unit": spec["resampling_unit"], "patient_identity_verified": spec["resampling_unit"] == "patient",
            "class_names": spec["class_names"], "draw_plan_sha256": plan.sha256,
            "support_sampling": spec.get("support_sampling", spec.get("sampling_policy", {})),
            "support_count_by_seed": {str(s): len(spec["seeds"][str(s)]["support_rows"]) for s in SEEDS},
            "support_group_count_by_seed": {str(s): len({str(r["group_id"]) for r in spec["seeds"][str(s)]["support_rows"]}) for s in SEEDS},
            "models": {}}
        summary["datasets"][dataset] = dataset_summary
        for model in MODELS:
            job = args.run / f"{model}_{dataset}"
            old_data, old_provenance = historical_predictions(model, dataset, exp)
            print(f"BOOTSTRAP historical {model}/{dataset}", flush=True)
            old_result = process(old_data, exp, spec, plan, args.run / "bootstrap" / f"{model}_{dataset}" / "historical_support", old_provenance)
            audits.append(old_result.summary["independent_sklearn_point_metric_max_abs_error"])
            history_rows.extend(per_seed_rows(model, dataset, "historical", old_result, spec["resampling_unit"]))
            fitted, spatial_fitted, reference = [], [], None
            for seed in SEEDS:
                path = job / f"seed_{seed}/predictions.npz"
                data = load_npz(path)
                verify_predictions(data, exp)
                if reference is None:
                    reference = data
                for key in ("original", "tta_center_original"):
                    if not np.array_equal(data[key], reference[key]):
                        raise ValueError(f"Nonadaptive {model}/{dataset}/{key} changed with seed")
                    if not np.allclose(data[key], old_data[key], rtol=1e-5, atol=2e-6):
                        raise ValueError(f"Historical baseline mismatch: {model}/{dataset}/{key}")
                print(f"BOOTSTRAP {model}/{dataset}/{seed}", flush=True)
                provenance = {"predictions_path": str(path), "predictions_sha256": sha(path),
                    "support_seed": seed, "dataset_protocol_sha256": source[str(Path(catalog["datasets"][dataset]["path"]))]}
                result = process(data, exp, spec, plan, args.run / "bootstrap" / f"{model}_{dataset}" / f"seed_{seed}", provenance)
                fitted.append(result)
                audits.append(result.summary["independent_sklearn_point_metric_max_abs_error"])
                per_seed.extend(per_seed_rows(model, dataset, seed, result, spec["resampling_unit"]))
                spatial_path = job / f"seed_{seed}/spatial.npz"
                if model == "AlphaRAD" and dataset == "chestxdet10":
                    spatial = load_npz(spatial_path)
                    frozen_spatial = spec["spatial_manifest"]
                    if sha(frozen_spatial["path"]) != frozen_spatial["sha256"]:
                        raise ValueError("Spatial holdout manifest changed")
                    frozen_spatial_rows = json.loads(Path(frozen_spatial["path"]).read_text())["rows"]
                    if [r["image"] for r in frozen_spatial_rows] != spatial["image_paths"].tolist():
                        # Inference may preserve classification order rather than
                        # box-annotation order; the complete identity set is fixed.
                        if set(r["image"] for r in frozen_spatial_rows) != set(spatial["image_paths"].tolist()):
                            raise ValueError("Spatial evaluation does not cover the frozen holdout")
                    point, draw, spatial_summary = spatial_bootstrap(spatial, exp, plan, spec["resampling_unit"])
                    native_spatial = json.loads((spatial_path.parent / "spatial_result.json").read_text())
                    for ib, branch in enumerate(BRANCHES):
                        if abs(point[ib].mean() - native_spatial["branches"][branch]["pointing_game"]) > 1e-12:
                            raise ValueError("Independent PG aggregation disagrees with native worker result")
                    spatial_summary["provenance"] = {"path": str(spatial_path), "sha256": sha(spatial_path)}
                    spatial_fitted.append((point, draw, spatial_summary))
                    folder = args.run / "bootstrap" / f"{model}_{dataset}" / f"seed_{seed}"
                    write_json(folder / "spatial_bootstrap_summary.json", spatial_summary)
                    np.savez_compressed(folder / "spatial_bootstrap_metrics.npz", point_metrics=point, replicate_metrics=draw,
                        branch_names=np.asarray(BRANCHES), class_names=exp["class_names"], draw_plan_sha256=np.asarray(plan.sha256))
                    for ib, branch in enumerate(BRANCHES):
                        ci = summarize_uncertainty(point[ib].mean() - point[0].mean(), draw[:, ib].mean(axis=1) - draw[:, 0].mean(axis=1))
                        spatial_seed_rows.append({"model": model, "dataset": dataset, "seed": seed, "branch": branch,
                            "pg_percent": float(100 * point[ib].mean()), "pg_delta_pp": ci["point"],
                            "pg_ci_low_pp": None if ci["ci"] is None else ci["ci"][0], "pg_ci_high_pp": None if ci["ci"] is None else ci["ci"][1],
                            "pg_valid_fraction": ci["valid_fraction"], "resampling_unit": spec["resampling_unit"]})
            branches, rows, by_class = aggregate_classification(model, dataset, fitted, spec["resampling_unit"])
            aggregate_rows.extend(rows)
            class_rows.extend(by_class)
            dataset_summary["test_counts"] = fitted[0].summary["test_counts"]
            dataset_summary["models"][model] = {"branches": branches}
            if spatial_fitted:
                p = np.stack([x[0] for x in spatial_fitted]).mean(axis=-1)
                d = np.stack([x[1] for x in spatial_fitted]).mean(axis=-1)
                spatial_summary = {"images": spatial_fitted[0][2]["images"], "groups": spatial_fitted[0][2]["groups"],
                    "resampling_unit": spec["resampling_unit"], "cases_per_class": spatial_fitted[0][2]["cases_per_class"], "branches": {}}
                for ib, branch in enumerate(BRANCHES):
                    gains = p[:, ib] - p[:, 0]
                    ci = summarize_uncertainty(gains.mean(), (d[:, :, ib] - d[:, :, 0]).mean(axis=0))
                    record = {"model": model, "dataset": dataset, "branch": branch, "resampling_unit": spec["resampling_unit"],
                        "pg_mean_percent": float(100 * p[:, ib].mean()), "pg_sd_pp": float(100 * p[:, ib].std(ddof=1)),
                        "pg_delta_pp": ci["point"], "pg_ci_low_pp": None if ci["ci"] is None else ci["ci"][0],
                        "pg_ci_high_pp": None if ci["ci"] is None else ci["ci"][1], "pg_valid_fraction": ci["valid_fraction"],
                        "pg_positive_seeds": int((gains > 0).sum())}
                    spatial_rows.append(record)
                    spatial_summary["branches"][branch] = {**record, "mean_gain_paired_cluster_bootstrap": ci}
                dataset_summary["models"][model]["spatial"] = spatial_summary
            write_json(args.run / "RESULTS.json", summary)
    if len(per_seed) != 80 or len(aggregate_rows) != 16 or len(spatial_seed_rows) != 20 or len(spatial_rows) != 4:
        raise ValueError("Incomplete required classification/spatial result coverage")
    write_csv(args.run / "PER_SEED_RESULTS.csv", per_seed)
    write_csv(args.run / "MEAN_SD_COMPARISON.csv", aggregate_rows)
    write_csv(args.run / "PER_CLASS_MEAN_SD.csv", class_rows)
    write_csv(args.run / "HISTORICAL_SUPPORT_CI.csv", history_rows)
    write_csv(args.run / "SPATIAL_PER_SEED.csv", spatial_seed_rows)
    write_csv(args.run / "SPATIAL_MEAN_SD.csv", spatial_rows)
    lines = ["RSNA / ChestXDet10：固定测试集、五支持抽样种子与配对置信区间", "",
        "BiomedCLIP、AlphaRAD × RSNA、ChestXDet10；种子42—46；每次 Original/Q/V/Q+V。",
        "η=0.3，中心裁剪80%；测试集、查询更新规则、双视图融合规则固定，不按测试成绩选择种子。",
        "RSNA按患者重采样；ChestXDet10没有可核实患者ID，按图像/精确重复组重采样，不能称患者级CI。",
        "每数据集2000次有放回配对bootstrap，seed20261024，同一抽样计划跨模型、支持种子、分支共用。",
        "指标为图像级分类、类别等权宏平均；AUPRC为PR曲线梯形面积；AP另列CSV/JSON。",
        "任一固定类别缺阳性或阴性的bootstrap重复令macro无效；不使用nanmean更换类别集合，报告有效率。",
        "均值±样本标准差(ddof=1)来自五支持种子。CI为给定五组支持样本时，平均增益的测试组抽样不确定性。",
        "不把五次测试预测拼接成五倍样本；CI未包含支持总体抽样不确定性；未做多重检验校正。",
        "这些测试集已用于之前方法开发；本轮不是未接触过的独立验证。历史支持另列，不混入五种子均值。", "",
        "| 模型 | 数据集 | 分支 | AUROC 均值±SD | ΔAUROC [95%CI] | 提升次数 | AUPRC 均值±SD | ΔAUPRC [95%CI] | 提升次数 | 两项同升 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        values = []
        for metric in METRIC_NAMES[:2]:
            values += [f"{row[metric+'_mean_percent']:.2f}±{row[metric+'_sd_pp']:.2f}", interval_text(row, metric), f"{row[metric+'_positive_seeds']}/5"]
        lines.append("| " + " | ".join([row["model"], row["dataset"], DISPLAY[row["branch"]], *values, f"{row['both_primary_positive_seeds']}/5"]) + " |")
    lines += ["", "AlphaRAD / ChestXDet10 定位：PG按各类别命中率再等权平均；不选择分割阈值。", "",
        "| 分支 | PG均值±SD | ΔPG [95%CI] | 提升次数 | 有效bootstrap比例 |", "|---|---:|---:|---:|---:|"]
    for row in spatial_rows:
        lines.append(f"| {DISPLAY[row['branch']]} | {row['pg_mean_percent']:.2f}±{row['pg_sd_pp']:.2f} | {interval_text(row, 'pg')} | {row['pg_positive_seeds']}/5 | {row['pg_valid_fraction']:.3%} |")
    lines += ["", "测试类别及稀有阳性统计（正负组可能重叠；RSNA为患者、Chest为图像/精确重复组）："]
    for dataset in DATASETS:
        info = summary["datasets"][dataset]
        lines.append(f"{dataset}: {info['test_images']}图 / {info['test_groups']}组；unit={info['resampling_unit']}")
        for name, counts in info["test_counts"].items():
            group_key = "patients_with_positive_image" if info["resampling_unit"] == "patient" else "groups_with_positive_image"
            lines.append(f"  {name}: 阳性图={counts['positive_images']}, 阴性图={counts['negative_images']}, 有阳性图的组={counts[group_key]}")
    lines += ["", "每种子80行：PER_SEED_RESULTS.csv；均值±SD16行：MEAN_SD_COMPARISON.csv。",
        "类别明细：PER_CLASS_MEAN_SD.csv及bootstrap/<模型_数据集>/seed_*/bootstrap_summary.json。",
        "历史支持配对CI单独在HISTORICAL_SUPPORT_CI.csv；历史支持不并入本轮五种子均值。",
        "空间逐种子/汇总：SPATIAL_PER_SEED.csv、SPATIAL_MEAN_SD.csv。",
        "实际组抽样倍数：*_cluster_draws.npz；bootstrap目录记录每次指标及配对增益CI。"]
    (args.run / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    analysis = ["结果解读：五支持种子，固定测试集", "",
        "以下报告所有组合，不按结果择优。正增益次数描述支持抽样敏感性；CI描述给定五组支持的测试组抽样不确定性。"]
    for row in aggregate_rows:
        if row["branch"] != "tta_center_refined":
            continue
        analysis.append(f"{row['model']} / {row['dataset']}: Q+V ΔAUROC {interval_text(row, 'auroc')}；ΔAUPRC {interval_text(row, 'auprc')}；两项同升 {row['both_primary_positive_seeds']}/5。")
    analysis += ["", "只有当区间完全高于0时，该未经多重检验校正的配对区间支持本次指标提升；跨0表示不确定，完全低于0支持下降。",
        "ChestXDet10无患者身份信息，重复组去重不保证同患者隔离；不能把其图像组CI解释成患者级CI。",
        "多标签支持抽样同时改变共病比例；支持候选池规模、覆盖和标签数应结合协议审计解释。",
        "本轮固定η和双视图，未调整方法；所有测试已参与历史开发，结果不等于新医院独立验证。"]
    (args.run / "ANALYSIS.txt").write_text("\n".join(analysis) + "\n")
    summary["status"] = "complete"
    summary["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    summary["coverage"] = {"classification_seed_rows": len(per_seed), "classification_mean_rows": len(aggregate_rows),
        "spatial_seed_rows": len(spatial_seed_rows), "spatial_mean_rows": len(spatial_rows)}
    summary["independent_sklearn_point_metric_max_abs_error"] = max(audits)
    write_json(args.run / "RESULTS.json", summary)
    artifacts = [args.run / name for name in ("RESULTS.json", "RESULTS.txt", "ANALYSIS.txt", "PER_SEED_RESULTS.csv",
        "MEAN_SD_COMPARISON.csv", "PER_CLASS_MEAN_SD.csv", "HISTORICAL_SUPPORT_CI.csv", "SPATIAL_PER_SEED.csv", "SPATIAL_MEAN_SD.csv")]
    write_json(args.run / "SUMMARY_AUDIT.json", {"status": "passed", "source_sha256": source,
        "artifact_sha256": {str(p): sha(p) for p in artifacts}, "coverage": summary["coverage"],
        "independent_sklearn_point_metric_max_abs_error": max(audits)})
    print(json.dumps({"status": "complete", "run": str(args.run), "coverage": summary["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
