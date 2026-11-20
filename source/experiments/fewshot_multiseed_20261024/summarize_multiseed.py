#!/usr/bin/env python3
"""Audit and summarize eleven methods on five fixed support draws.

The same dataset-specific group draw is used across models, methods and support
seeds. Metrics are computed separately for each seed, then averaged; predictions
are never averaged or concatenated. Intervals condition on the five support sets.
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
ENGINE = ROOT.parent / "chexchonet_stability_20261024" / "cluster_bootstrap.py"
sys.path.insert(0, str(ENGINE.parent))
from cluster_bootstrap import (BootstrapResult, METRIC_NAMES, interval_summary,
                               make_patient_draw_plan, paired_cluster_bootstrap)

MODELS = ("BiomedCLIP", "AlphaRAD")
DATASETS = ("rsna_pneumonia", "chestxdet10")
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("original", "positive_query", "query", "view", "query_view",
           "prototype_score_only", "prototype_score_mix_oof", "tip_adapter_ovr_oof",
           "gda_ovr_ks_oof", "class_local_difference_score_oof", "simnl_feature_ovr")
NATIVE_METHODS = METHODS[:5]
DISPLAY = dict(zip(METHODS, ("Original", "Positive query", "Q", "V", "Q+V",
    "Prototype", "Prototype score mix", "Tip-Adapter OvR", "GDA KS OvR",
    "Class-local difference score", "SimNL feature OvR")))
EXPECTED_COHORTS = {"rsna_pneumonia": (5327, 3720, "patient"),
                    "chestxdet10": (492, 492, "image_exact_duplicate_group")}
UNCERTAINTY_NOTE = (
    "Sample SD (ddof=1) describes sensitivity across five support seeds. Paired "
    "95% percentile intervals describe test-group sampling uncertainty conditional "
    "on these five fixed support sets. Each replicate averages the five seed-specific "
    "metrics/differences; predictions are never averaged or concatenated. These are "
    "not joint support-population/test-population intervals. No multiple-comparison "
    "adjustment. All predeclared comparisons are retained; no best test competitor "
    "or support seed is selected.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def frozen_json(entry):
    path = Path(entry["path"])
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Frozen file SHA256 mismatch: {path}")
    return json.loads(path.read_text())


def prepare_expected(spec):
    rows = spec["test_rows"]
    classes = spec["class_names"]
    expected = {"image_paths": np.asarray([r["image"] for r in rows]),
                "group_ids": np.asarray([str(r["group_id"]) for r in rows]),
                "labels": np.asarray([r["labels"] for r in rows]),
                "class_names": np.asarray(classes)}
    if len(set(expected["image_paths"].tolist())) != len(rows):
        raise ValueError("Duplicate fixed-test image path")
    if len(set(classes)) != len(classes) or expected["labels"].shape != (len(rows), len(classes)):
        raise ValueError("Invalid test class names or label shape")
    labels = expected["labels"]
    if not np.isin(labels, (0, 1)).all() or np.any(labels.sum(0) == 0) or np.any(labels.sum(0) == len(rows)):
        raise ValueError("Fixed-test labels must be binary with evaluable classes")
    if spec["resampling_unit"] == "patient":
        patient_to_group, group_to_patient = {}, {}
        for row in rows:
            patient, group = str(row.get("patient_id", "")), str(row["group_id"])
            if patient in ("", "None", "nan"):
                raise ValueError("Verified patient ID missing")
            if patient_to_group.setdefault(patient, group) != group or group_to_patient.setdefault(group, patient) != patient:
                raise ValueError("RSNA patient and bootstrap identities are not one-to-one")
    elif spec["resampling_unit"] != "image_exact_duplicate_group":
        raise ValueError("Unsupported resampling unit")
    frozen_rows = frozen_json(spec["test_manifest"])["rows"]
    if [r["image"] for r in frozen_rows] != expected["image_paths"].tolist():
        raise ValueError("Frozen manifest and protocol identities/order differ")
    if not np.array_equal([r["labels"] for r in frozen_rows], labels):
        raise ValueError("Frozen manifest and protocol labels differ")
    return expected


def verify_predictions(data, expected, methods=METHODS):
    for key in ("image_paths", "class_names"):
        if key not in data or not np.array_equal(data[key], expected[key]):
            raise ValueError(f"Prediction/protocol mismatch: {key}")
    for key in ("labels", "group_ids"):
        if key in data and not np.array_equal(data[key], expected[key]):
            raise ValueError(f"Prediction/protocol mismatch: {key}")
    for method in methods:
        if method not in data or data[method].shape != expected["labels"].shape or not np.isfinite(data[method]).all():
            raise ValueError(f"Invalid scores: {method}")
    # Comparator margins need not be probabilities; ranking metrics accept both.


def metric_audit(labels, scores, point_metrics, methods=METHODS):
    maximum = 0.0
    for imethod, method in enumerate(methods):
        for ic in range(labels.shape[1]):
            y, score = labels[:, ic], scores[method][:, ic]
            precision, recall, _ = precision_recall_curve(y, score)
            expected = np.asarray((roc_auc_score(y, score), auc(recall, precision),
                                   average_precision_score(y, score)))
            maximum = max(maximum, float(np.max(np.abs(expected - point_metrics[imethod, ic]))))
    if maximum > 1e-12:
        raise ValueError(f"Independent sklearn point-metric audit failed: {maximum}")
    return maximum


def neutral_group_language(value):
    if isinstance(value, dict):
        return {key.replace("patients", "groups").replace("patient", "group"):
                neutral_group_language(item) for key, item in value.items()}
    if isinstance(value, list):
        return [neutral_group_language(item) for item in value]
    if isinstance(value, str):
        return value.replace("patients", "groups").replace("patient", "group")
    return value


def save_group_plan(plan, path, unit):
    np.savez_compressed(path, group_ids=plan.patient_ids, counts=plan.counts,
        seed=np.asarray(plan.seed), sha256=np.asarray(plan.sha256),
        resampling_unit=np.asarray(unit), generator=np.asarray("numpy.Generator(PCG64)"))


def process_classification(data, expected, spec, plan, folder, provenance, native_result):
    verify_predictions(data, expected)
    summary_path, archive_path = folder / "bootstrap_summary.json", folder / "bootstrap_metrics.npz"
    if summary_path.exists() and archive_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("provenance") != provenance or summary["draw_plan_sha256"] != plan.sha256:
            raise ValueError(f"Bootstrap cache provenance changed; use a fresh cache directory: {folder}")
        if sha(archive_path) != summary["bootstrap_metrics_sha256"]:
            raise ValueError(f"Bootstrap metric cache hash mismatch: {archive_path}")
        arrays = load_npz(archive_path)
        result = BootstrapResult(summary, arrays["point_metrics"], arrays["replicate_metrics"],
            tuple(arrays["branch_names"]), tuple(arrays["task_names"]), tuple(arrays["metric_names"]), plan)
    else:
        result = paired_cluster_bootstrap(expected["labels"], {m: data[m] for m in METHODS},
            expected["group_ids"], task_names=spec["class_names"], plan=plan)
        if spec["resampling_unit"] != "patient":
            result.summary = neutral_group_language(result.summary)
        result.summary.update(resampling_unit=spec["resampling_unit"], provenance=provenance,
            patient_identity_verified=spec["resampling_unit"] == "patient",
            uncertainty_scope="test-group sampling conditional on this seed's fixed support set",
            independent_sklearn_point_metric_max_abs_error=metric_audit(
                expected["labels"], data, result.point_metrics))
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, point_metrics=result.point_metrics,
            replicate_metrics=result.replicate_metrics, branch_names=np.asarray(METHODS),
            task_names=np.asarray(spec["class_names"]), metric_names=np.asarray(METRIC_NAMES),
            draw_plan_sha256=np.asarray(plan.sha256))
        result.summary["bootstrap_metrics_sha256"] = sha(archive_path)
        write_json(summary_path, result.summary)
    for imethod, method in enumerate(METHODS):
        metrics = native_result["methods"][method]["metrics"]
        for im, key in enumerate(("macro_auroc", "macro_auprc")):
            recorded = metrics[key]
            if recorded is None or abs(float(recorded) - result.point_metrics[imethod, :, im].mean()) > 1e-10:
                raise ValueError(f"Native result and audited metric disagree: {method}/{key}")
    return result


def percent_interval(point, draws):
    result = interval_summary(float(point), np.asarray(draws, dtype=np.float64))
    result["point"] = None if result["point"] is None else 100 * result["point"]
    result["ci"] = None if result["ci"] is None else [100 * x for x in result["ci"]]
    result["unit"] = "percentage_points"
    return result


def interval_columns(prefix, value):
    return {prefix + "_ci_low_pp": None if value["ci"] is None else value["ci"][0],
            prefix + "_ci_high_pp": None if value["ci"] is None else value["ci"][1],
            prefix + "_valid_replicates": value["valid_replicates"],
            prefix + "_invalid_replicates": value["invalid_replicates"],
            prefix + "_valid_fraction": value["valid_fraction"]}


def comparison_pairs(methods=METHODS):
    """Predeclared directed pairs; keep every comparator, deduplicate originals."""
    pairs = [(method, "original") for method in methods if method != "original"]
    pairs += [(target, reference) for target in ("query", "query_view") if target in methods
              for reference in methods if reference != target]
    return tuple(dict.fromkeys(pairs))


def paired_mean_summary(points, draws, target_index, reference_index, seeds=SEEDS):
    """Input axes: points[seed], draws[seed, replicate], plus method as last axis."""
    if points.shape[0] != len(seeds) or draws.shape[0] != len(seeds):
        raise ValueError("Support seed axis differs from declared seeds")
    gains = points[:, target_index] - points[:, reference_index]
    replicate_gains = (draws[:, :, target_index] - draws[:, :, reference_index]).mean(axis=0)
    interval = percent_interval(gains.mean(), replicate_gains)
    return {"mean_gain_pp": float(100 * gains.mean()), "gain_sd_pp": float(100 * gains.std(ddof=1)),
        "minimum_gain_pp": float(100 * gains.min()), "maximum_gain_pp": float(100 * gains.max()),
        "wins": int((gains > 0).sum()), "ties": int((gains == 0).sum()), "losses": int((gains < 0).sum()),
        "seed_gain_pp": {str(seed): float(100 * value) for seed, value in zip(seeds, gains)},
        "mean_gain_paired_cluster_bootstrap": interval}


def summarize_arrays(model, dataset, points, draws, class_names, unit,
                     methods=METHODS, metric_names=METRIC_NAMES):
    """Aggregate seed/method/class metrics, preserving the paired draw axis.

    points: [seed, method, class, metric]
    draws:  [seed, replicate, method, class, metric]
    """
    if points.shape != (len(SEEDS), len(methods), len(class_names), len(metric_names)):
        raise ValueError("Unexpected point-metric axes")
    if draws.shape[0] != len(SEEDS) or draws.shape[2:] != points.shape[1:]:
        raise ValueError("Unexpected bootstrap-metric axes")
    # A missing class makes the fixed-class macro invalid. Never nanmean.
    macro_points, macro_draws = points.mean(axis=2), draws.mean(axis=3)
    per_seed, means, per_class, comparisons, seed_comparisons, class_comparisons = [], [], [], [], [], []
    detail = {"methods": {}, "comparisons": {}}
    base = dict(model=model, dataset=dataset, resampling_unit=unit)
    original = methods.index("original")
    for ib, method in enumerate(methods):
        method_detail = {"macro": {}, "per_class": {}}
        detail["methods"][method] = method_detail
        row = dict(base, method=method, seed_count=len(SEEDS))
        for im, metric in enumerate(metric_names):
            values = macro_points[:, ib, im]
            absolute_ci = percent_interval(values.mean(), macro_draws[:, :, ib, im].mean(axis=0))
            comparison = paired_mean_summary(macro_points[:, :, im], macro_draws[:, :, :, im], ib, original)
            item = {"mean_percent": float(100 * values.mean()),
                "support_seed_sd_pp": float(100 * values.std(ddof=1)),
                "seed_metric_percent": {str(seed): float(100 * v) for seed, v in zip(SEEDS, values)},
                "mean_metric_cluster_bootstrap": absolute_ci, "vs_original": comparison}
            method_detail["macro"][metric] = item
            row.update({metric + "_mean_percent": item["mean_percent"], metric + "_sd_pp": item["support_seed_sd_pp"],
                metric + "_delta_pp": comparison["mean_gain_pp"], metric + "_positive_seeds": comparison["wins"],
                **interval_columns(metric, comparison["mean_gain_paired_cluster_bootstrap"]),
                **interval_columns(metric + "_absolute", absolute_ci)})
        if len(metric_names) >= 2:
            row["both_primary_positive_seeds"] = int(np.all(
                macro_points[:, ib, :2] - macro_points[:, original, :2] > 0, axis=1).sum())
        means.append(row)
        for iseed, seed in enumerate(SEEDS):
            seed_row = dict(base, method=method, seed=seed)
            for im, metric in enumerate(metric_names):
                point = macro_points[iseed, ib, im]
                ci = percent_interval(point - macro_points[iseed, original, im],
                    macro_draws[iseed, :, ib, im] - macro_draws[iseed, :, original, im])
                seed_row.update({metric + "_percent": float(100 * point), metric + "_delta_pp": ci["point"],
                                 **interval_columns(metric, ci)})
            per_seed.append(seed_row)
        for ic, class_name in enumerate(class_names):
            class_row = dict(base, method=method, class_name=class_name)
            class_detail = {}
            method_detail["per_class"][class_name] = class_detail
            for im, metric in enumerate(metric_names):
                values = points[:, ib, ic, im]
                comparison = paired_mean_summary(points[:, :, ic, im], draws[:, :, :, ic, im], ib, original)
                class_detail[metric] = {"mean_percent": float(100 * values.mean()),
                    "support_seed_sd_pp": float(100 * values.std(ddof=1)), "vs_original": comparison}
                class_row.update({metric + "_mean_percent": float(100 * values.mean()),
                    metric + "_sd_pp": float(100 * values.std(ddof=1)), metric + "_delta_pp": comparison["mean_gain_pp"],
                    metric + "_positive_seeds": comparison["wins"],
                    **interval_columns(metric, comparison["mean_gain_paired_cluster_bootstrap"])})
            per_class.append(class_row)
    for target, reference in comparison_pairs(methods):
        ti, ri = methods.index(target), methods.index(reference)
        pair_id = target + "__minus__" + reference
        pair_detail = {"target": target, "reference": reference, "macro": {}, "per_class": {}}
        detail["comparisons"][pair_id] = pair_detail
        row = dict(base, target=target, reference=reference, seed_count=len(SEEDS))
        for im, metric in enumerate(metric_names):
            item = paired_mean_summary(macro_points[:, :, im], macro_draws[:, :, :, im], ti, ri)
            pair_detail["macro"][metric] = item
            row.update({metric + "_delta_pp": item["mean_gain_pp"], metric + "_gain_sd_pp": item["gain_sd_pp"],
                metric + "_wins": item["wins"], metric + "_ties": item["ties"], metric + "_losses": item["losses"],
                **interval_columns(metric, item["mean_gain_paired_cluster_bootstrap"]),
                **{metric + "_seed_" + seed + "_gain_pp": value for seed, value in item["seed_gain_pp"].items()}})
        if len(metric_names) >= 2:
            row["both_primary_positive_seeds"] = int(np.all(
                macro_points[:, ti, :2] - macro_points[:, ri, :2] > 0, axis=1).sum())
        comparisons.append(row)
        for iseed, seed in enumerate(SEEDS):
            seed_row = dict(base, target=target, reference=reference, seed=seed)
            for im, metric in enumerate(metric_names):
                gain = macro_points[iseed, ti, im] - macro_points[iseed, ri, im]
                ci = percent_interval(gain, macro_draws[iseed, :, ti, im] - macro_draws[iseed, :, ri, im])
                seed_row.update({metric + "_gain_pp": float(100 * gain), metric + "_win": int(gain > 0),
                                 **interval_columns(metric, ci)})
            seed_comparisons.append(seed_row)
        for ic, class_name in enumerate(class_names):
            class_row = dict(base, target=target, reference=reference, class_name=class_name)
            pair_detail["per_class"][class_name] = {}
            for im, metric in enumerate(metric_names):
                item = paired_mean_summary(points[:, :, ic, im], draws[:, :, :, ic, im], ti, ri)
                pair_detail["per_class"][class_name][metric] = item
                class_row.update({metric + "_delta_pp": item["mean_gain_pp"], metric + "_wins": item["wins"],
                                  **interval_columns(metric, item["mean_gain_paired_cluster_bootstrap"])})
            class_comparisons.append(class_row)
    return {"detail": detail, "per_seed": per_seed, "means": means, "per_class": per_class,
            "comparisons": comparisons, "seed_comparisons": seed_comparisons,
            "class_comparisons": class_comparisons}


def spatial_bootstrap(data, expected, plan, frozen_spatial_rows, native_result):
    paths = data["image_paths"].tolist()
    lookup = {path: i for i, path in enumerate(expected["image_paths"].tolist())}
    if len(paths) != len(set(paths)) or set(paths) != {row["image"] for row in frozen_spatial_rows}:
        raise ValueError("PG identities differ from the complete frozen spatial holdout")
    if any(path not in lookup for path in paths):
        raise ValueError("PG identity outside the fixed classification test")
    indices = np.asarray([lookup[path] for path in paths])
    if not np.array_equal(data["group_ids"], expected["group_ids"][indices]):
        raise ValueError("PG and classification resampling groups differ")
    if not np.array_equal(data["class_names"], expected["class_names"]):
        raise ValueError("PG class order differs")
    inverse = np.searchsorted(plan.patient_ids, data["group_ids"])
    weights = plan.counts[:, inverse].astype(np.float64)
    points, draws, reference_cases = [], [], None
    for method in NATIVE_METHODS:
        hits, cases = data[method + "_hits"], data[method + "_cases"]
        if hits.shape != (len(paths), len(expected["class_names"])) or cases.shape != hits.shape:
            raise ValueError("Invalid PG hit/case axes")
        if not np.isin(hits, (0, 1)).all() or not np.isin(cases, (0, 1)).all() or np.any(hits > cases):
            raise ValueError("PG hits/cases must be binary and hits <= cases")
        if reference_cases is not None and not np.array_equal(reference_cases, cases):
            raise ValueError("PG case mask differs across methods")
        reference_cases = cases
        if np.any(cases.sum(0) == 0):
            raise ValueError("Empty fixed PG class")
        point = hits.sum(0) / cases.sum(0)
        denominators = weights @ cases
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled = (weights @ hits) / denominators
        sampled[denominators == 0] = np.nan
        if abs(point.mean() - native_result["spatial"][method]["pointing_game"]) > 1e-12:
            raise ValueError(f"Independent PG result mismatch: {method}")
        points.append(point)
        draws.append(sampled)
    metadata = {"images": len(paths), "groups": len(set(data["group_ids"].tolist())),
        "cases_per_class": reference_cases.sum(0).astype(int).tolist(), "draw_plan_sha256": plan.sha256,
        "metric_definition": "unweighted class mean of image-class hits/cases",
        "invalid_draw_policy": "a zero-case class invalidates the fixed-class macro",
        "sampling_scope": "reuse the entire classification-cohort group plan, including images without boxes",
        "role": "secondary native spatial endpoint; score-only comparators have no native spatial output"}
    return np.stack(points)[..., None], np.stack(draws, axis=1)[..., None], metadata


def number(value, signed=False):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def ci_text(row, metric):
    return (f"{number(row[metric + '_delta_pp'], True)} "
            f"[{number(row[metric + '_ci_low_pp'], True)}, {number(row[metric + '_ci_high_pp'], True)}]")


def write_report(run, summary, records):
    lines = ["RSNA / ChestXDet10 — eleven methods, five support seeds", "",
        "Seeds: 42–46. Fixed held-out tests: RSNA 5327 images / 3720 verified patients; "
        "ChestXDet10 492 images / 492 image/exact-duplicate groups.",
        "ChestXDet10 has no verified patient IDs; its intervals are not patient-level intervals.",
        f"Bootstrap: {summary['bootstrap_replicates']} paired group draws, seed {summary['bootstrap_seed']}; "
        "one plan per dataset shared across all models, methods and support seeds.",
        "Metrics: image-level class-macro AUROC, trapezoidal PR AUPRC; AP is also in CSV/JSON. "
        "All numbers below use percent or percentage points.", UNCERTAINTY_NOTE,
        "These test sets were exposed during previous exploratory method development; "
        "this is not independent external confirmation.", "",
        "| Model | Dataset | Method | AUROC mean±SD | Δ vs Original [95% CI] | Wins | AUPRC mean±SD | Δ vs Original [95% CI] | Wins |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in records["means"]:
        values = [row["model"], row["dataset"], DISPLAY[row["method"]]]
        for metric in METRIC_NAMES[:2]:
            values += [f"{row[metric + '_mean_percent']:.2f}±{row[metric + '_sd_pp']:.2f}",
                       ci_text(row, metric), f"{row[metric + '_positive_seeds']}/5"]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "All Q / Q+V comparisons; comparator order is fixed in advance.", "",
        "| Model | Dataset | Target | Comparator | ΔAUROC [95% CI] | Wins | ΔAUPRC [95% CI] | Wins |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for row in records["comparisons"]:
        if row["target"] not in ("query", "query_view"):
            continue
        lines.append("| " + " | ".join([row["model"], row["dataset"], DISPLAY[row["target"]],
            DISPLAY[row["reference"]], ci_text(row, "auroc"), f"{row['auroc_wins']}/5",
            ci_text(row, "auprc"), f"{row['auprc_wins']}/5"]) + " |")
    lines += ["", "AlphaRAD / ChestXDet10 native pointing game (secondary endpoint).", "",
        "| Method | PG mean±SD | ΔPG vs Original [95% CI] | Wins |",
        "|---|---:|---:|---:|"]
    for row in records["spatial_means"]:
        if row["status"] != "available":
            lines.append(f"| {DISPLAY[row['method']]} | NA | NA | NA |")
        else:
            lines.append(f"| {DISPLAY[row['method']]} | {row['pg_mean_percent']:.2f}±{row['pg_sd_pp']:.2f} | "
                         f"{ci_text(row, 'pg')} | {row['pg_positive_seeds']}/5 |")
    lines += ["", "Score-only comparators have no native localization output; their PG is NA, not zero.",
        "PER_SEED_RESULTS.csv: 220 classification method/seed rows. MEAN_SD_COMPARISON.csv: 44 rows.",
        "PAIRED_COMPARISONS.csv and PER_SEED_PAIRED_COMPARISONS.csv retain all predeclared comparisons.",
        "PER_CLASS_MEAN_SD.csv and PER_CLASS_PAIRED_COMPARISONS.csv contain class detail.",
        "SPATIAL_*.csv contain native PG results plus explicit NA for score-only methods.",
        "*_cluster_draws.npz and bootstrap/ record shared multiplicities and every replicate metric.",
        "RESULTS.json stores metric definitions, class/seed detail, provenance and invalid-replicate counts."]
    (run / "RESULTS.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol_v1/catalog.json")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.protocol.read_text())
    if (tuple(catalog["support_seeds"]) != SEEDS or tuple(catalog["models"]) != MODELS
            or tuple(catalog["methods"]) != METHODS):
        raise ValueError("Expected predeclared models, eleven methods and support seeds 42–46")
    sources = {str(Path(__file__)): sha(__file__), str(ENGINE): sha(ENGINE),
               str(args.protocol.resolve()): sha(args.protocol)}
    specs, expected, plans = {}, {}, {}
    for dataset in DATASETS:
        entry = catalog["datasets"][dataset]
        spec = frozen_json(entry)
        sources[str(Path(entry["path"]))] = entry["sha256"]
        if set(spec["seeds"]) != {str(seed) for seed in SEEDS}:
            raise ValueError("Dataset protocol lacks exactly the five support seeds")
        for seed in SEEDS:
            native_entry = spec["seeds"][str(seed)]["native_protocol"]
            frozen_json(native_entry)
            sources[str(Path(native_entry["path"]))] = native_entry["sha256"]
        exp = prepare_expected(spec)
        if (len(exp["labels"]), len(set(exp["group_ids"].tolist())), spec["resampling_unit"]) != EXPECTED_COHORTS[dataset]:
            raise ValueError("Fixed test cohort or grouping differs from the requested experiment")
        plan = make_patient_draw_plan(exp["group_ids"], n_resamples=args.replicates, seed=args.seed)
        save_group_plan(plan, args.run / f"{dataset}_cluster_draws.npz", spec["resampling_unit"])
        specs[dataset], expected[dataset], plans[dataset] = spec, exp, plan
    records = {key: [] for key in ("per_seed", "means", "per_class", "comparisons", "seed_comparisons",
        "class_comparisons", "spatial_per_seed", "spatial_means", "spatial_per_class", "spatial_comparisons")}
    summary = {"schema_version": 1, "status": "running",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "support_seeds": list(SEEDS), "methods": list(METHODS), "models": list(MODELS),
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "source_sha256": sources, "uncertainty_note": UNCERTAINTY_NOTE,
        "metric_definition": "image-level AUROC / trapezoidal PR AUPRC / AP, unweighted fixed-class macro",
        "invalid_draw_policy": "if any fixed class is invalid, the macro is invalid; never nanmean",
        "comparison_policy": [{"target": a, "reference": b} for a, b in comparison_pairs()],
        "test_exposure_note": "Tests were exposed in previous exploratory development; not independent external confirmation.",
        "datasets": {}}
    audits, input_hashes = [], {}
    for dataset in DATASETS:
        spec, exp, plan = specs[dataset], expected[dataset], plans[dataset]
        dataset_summary = {"test_images": len(exp["labels"]), "test_groups": len(plan.patient_ids),
            "resampling_unit": spec["resampling_unit"], "patient_identity_verified": spec["resampling_unit"] == "patient",
            "class_names": spec["class_names"], "draw_plan_sha256": plan.sha256,
            "support_count_by_seed": {str(seed): len(spec["seeds"][str(seed)]["support_rows"]) for seed in SEEDS},
            "models": {}}
        summary["datasets"][dataset] = dataset_summary
        for model in MODELS:
            fitted, spatial, reference = [], [], None
            for seed in SEEDS:
                folder = args.run / f"{model}_{dataset}" / "evaluation" / f"seed_{seed}"
                native_result = json.loads((folder / "result.json").read_text())
                if native_result.get("status") != "complete" or native_result.get("support_seed") != seed:
                    raise ValueError(f"Incomplete evaluation or wrong support seed: {folder}")
                if native_result.get("model") != model or native_result.get("dataset") != dataset:
                    raise ValueError(f"Wrong evaluation model/dataset: {folder}")
                native_entry = spec["seeds"][str(seed)]["native_protocol"]
                if native_result.get("protocol_sha256") != native_entry["sha256"]:
                    raise ValueError(f"Evaluation used a different support protocol: {folder}")
                if set(native_result["methods"]) != set(METHODS):
                    raise ValueError(f"Expected exactly eleven predeclared methods: {folder}")
                for filename, key in (("fit_info.json", "fit_info_sha256"), ("predictions.npz", "predictions_sha256")):
                    if sha(folder / filename) != native_result[key]:
                        raise ValueError(f"Evaluation artifact changed: {folder / filename}")
                data = load_npz(folder / "predictions.npz")
                verify_predictions(data, exp)
                if reference is None:
                    reference = data
                for method in ("original", "view"):
                    if not np.array_equal(data[method], reference[method]):
                        raise ValueError(f"Nonadaptive {method} changed across support seeds: {model}/{dataset}")
                provenance = {"support_seed": seed, "source_sha256": sources,
                    "input_sha256": {str(folder / name): sha(folder / name)
                        for name in ("result.json", "predictions.npz", "fit_info.json")}}
                input_hashes.update(provenance["input_sha256"])
                out = args.run / "bootstrap" / f"{model}_{dataset}" / f"seed_{seed}"
                print(f"BOOTSTRAP {model}/{dataset}/seed_{seed}: eleven methods", flush=True)
                result = process_classification(data, exp, spec, plan, out, provenance, native_result)
                fitted.append(result)
                audits.append(result.summary["independent_sklearn_point_metric_max_abs_error"])
                if model == "AlphaRAD" and dataset == "chestxdet10":
                    spatial_path = folder / "spatial.npz"
                    if sha(spatial_path) != native_result['spatial_sha256']:
                        raise ValueError(f"Spatial predictions changed after evaluation: {spatial_path}")
                    spatial_data = load_npz(spatial_path)
                    frozen_rows = frozen_json(spec["spatial_manifest"])["rows"]
                    point, draw, metadata = spatial_bootstrap(spatial_data, exp, plan, frozen_rows, native_result)
                    metadata["input_sha256"] = {str(spatial_path): sha(spatial_path)}
                    input_hashes.update(metadata["input_sha256"])
                    np.savez_compressed(out / "spatial_bootstrap_metrics.npz", point_metrics=point,
                        replicate_metrics=draw, method_names=np.asarray(NATIVE_METHODS),
                        class_names=exp["class_names"], metric_names=np.asarray(["pg"]),
                        draw_plan_sha256=np.asarray(plan.sha256))
                    write_json(out / "spatial_bootstrap_summary.json", metadata)
                    spatial.append((point, draw, metadata))
            aggregated = summarize_arrays(model, dataset, np.stack([r.point_metrics for r in fitted]),
                np.stack([r.replicate_metrics for r in fitted]), spec["class_names"], spec["resampling_unit"])
            dataset_summary["test_counts"] = fitted[0].summary["test_counts"]
            dataset_summary["models"][model] = aggregated.pop("detail")
            for key, rows in aggregated.items():
                records[key].extend(rows)
            if spatial:
                spatial_summary = summarize_arrays(model, dataset, np.stack([s[0] for s in spatial]),
                    np.stack([s[1] for s in spatial]), spec["class_names"], spec["resampling_unit"],
                    NATIVE_METHODS, ("pg",))
                dataset_summary["models"][model]["spatial"] = {**spatial[0][2], **spatial_summary["detail"],
                    "method_availability": {m: "native" if m in NATIVE_METHODS else "NA: no native spatial output" for m in METHODS}}
                for key in ("per_seed", "means", "per_class", "comparisons"):
                    records["spatial_" + key].extend(dict(row, status="available") for row in spatial_summary[key])
                for method in METHODS[len(NATIVE_METHODS):]:
                    base = dict(model=model, dataset=dataset, method=method, status="NA: no native spatial output",
                                resampling_unit=spec["resampling_unit"])
                    records["spatial_means"].append(base)
                    records["spatial_per_seed"].extend(dict(base, seed=seed) for seed in SEEDS)
            write_json(args.run / "RESULTS.json", summary)
    coverage = {key + "_rows": len(rows) for key, rows in records.items()}
    if (coverage["per_seed_rows"], coverage["means_rows"], coverage["comparisons_rows"],
        coverage["spatial_per_seed_rows"], coverage["spatial_means_rows"]) != (220, 44, 112, 55, 11):
        raise ValueError(f"Incomplete predeclared result coverage: {coverage}")
    filenames = {"per_seed": "PER_SEED_RESULTS.csv", "means": "MEAN_SD_COMPARISON.csv",
        "per_class": "PER_CLASS_MEAN_SD.csv", "comparisons": "PAIRED_COMPARISONS.csv",
        "seed_comparisons": "PER_SEED_PAIRED_COMPARISONS.csv", "class_comparisons": "PER_CLASS_PAIRED_COMPARISONS.csv",
        "spatial_per_seed": "SPATIAL_PER_SEED.csv", "spatial_means": "SPATIAL_MEAN_SD.csv",
        "spatial_per_class": "SPATIAL_PER_CLASS_MEAN_SD.csv", "spatial_comparisons": "SPATIAL_PAIRED_COMPARISONS.csv"}
    for key, filename in filenames.items():
        write_csv(args.run / filename, records[key])
    summary.update(status="complete", completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        coverage=coverage, input_sha256=input_hashes, independent_sklearn_point_metric_max_abs_error=max(audits))
    write_json(args.run / "RESULTS.json", summary)
    write_report(args.run, summary, records)
    artifacts = [args.run / name for name in (*filenames.values(), "RESULTS.json", "RESULTS.txt")]
    write_json(args.run / "SUMMARY_AUDIT.json", {"status": "passed", "source_sha256": sources,
        "input_sha256": input_hashes, "artifact_sha256": {str(p): sha(p) for p in artifacts},
        "coverage": coverage, "independent_sklearn_point_metric_max_abs_error": max(audits),
        "draw_plan_sha256": {d: plan.sha256 for d, plan in plans.items()}})
    print(json.dumps({"status": "complete", "run": str(args.run), "coverage": coverage}), flush=True)


if __name__ == "__main__":
    main()
