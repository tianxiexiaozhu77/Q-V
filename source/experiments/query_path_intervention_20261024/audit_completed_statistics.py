#!/usr/bin/env python3
"""Independent final audit; no model inference and no edits to archived results."""
from __future__ import annotations
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
RUN = ROOT / "runs/siim_holdout_20261024_v1"
STATS = RUN / "evaluation"
JOB = RUN / "AlphaRAD_siim_new_holdout"
SEEDS = (42, 43, 44, 45, 46)
METHODS = ("original", "readout_only", "aggregation_only", "query", "score_retained_original_map")
METRICS = ("auroc", "auprc", "average_precision")
SPATIAL = ("pixel_auroc", "mask_pg")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def metrics(y, score, weights=None):
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=weights)
    return np.asarray([roc_auc_score(y, score, sample_weight=weights), auc(recall, precision),
                       average_precision_score(y, score, sample_weight=weights)])


def difference_direction(features, labels):
    value = features[labels == 1].mean(0) - features[labels == 0].mean(0)
    length = np.linalg.norm(value)
    return value / length if length > 1e-12 else np.zeros_like(value)


def confirm_summary(points, draws, means, pairs, metric_names):
    maximum = 0.
    comparisons = []
    for method_index, method in enumerate(METHODS):
        row = next(row for row in means if row["method"] == method)
        for metric_index, metric in enumerate(metric_names):
            values = points[:, method_index, metric_index] * 100
            expected = [values.mean(), values.std(ddof=1)]
            observed = [float(row[metric + "_mean_percent"]), float(row[metric + "_sd_pp"])]
            maximum = max(maximum, float(np.max(np.abs(np.asarray(expected) - observed))))
            assert_allclose(observed, expected, atol=1e-10, rtol=0)
    for row in pairs:
        ti, ri = METHODS.index(row["target"]), METHODS.index(row["reference"])
        comparison = {"target": row["target"], "reference": row["reference"]}
        for metric_index, metric in enumerate(metric_names):
            point = (points[:, ti, metric_index] - points[:, ri, metric_index]).mean() * 100
            paired = (draws[:, :, ti, metric_index] - draws[:, :, ri, metric_index]).mean(0) * 100
            valid = paired[np.isfinite(paired)]
            low, high = np.quantile(valid, [.025, .975])
            actual = np.asarray([float(row[metric + key]) for key in ("_delta_pp", "_ci_low_pp", "_ci_high_pp")])
            expected = np.asarray([point, low, high])
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
            assert_allclose(actual, expected, atol=1e-10, rtol=0)
            comparison[metric] = {"difference_pp": float(point), "ci95_pp": [float(low), float(high)],
                                  "valid_replicates": len(valid)}
        comparisons.append(comparison)
    return {"maximum_error_pp": maximum, "comparisons": comparisons}


def check_supplement():
    build = read(ROOT / "SUPPLEMENT_BUILD_AUDIT.json")
    for path, digest in build["sources"].items():
        assert sha(path) == digest, path
    assert sha(build["report"]) == build["report_sha256"]
    assert sha(ROOT / "MANUSCRIPT_INSERT.md") == build["insert_sha256"]
    assert sha(ROOT / "build_supplement.py") == build["script_sha256"]
    text = Path(build["report"]).read_text()
    insert = (ROOT / "MANUSCRIPT_INSERT.md").read_text()
    names = {"rsna_pneumonia": "RSNA（既有开发集）", "chestxdet10": "ChestXDet10（既有开发集）",
             "siim_new_holdout": "SIIM（新增同源留出）"}
    display = dict(zip(METHODS, ("Original", "仅改读出 R", "仅改聚合 A", "完整 Q", "S＋保留 Original 空间图")))
    verified = []
    for run in (ROOT / "runs/diagnostic_20261024_v1", RUN):
        source = run / "evaluation"
        classification_means = rows(source / "MEAN_SD_COMPARISON.csv")
        classification_pairs = rows(source / "PAIRED_COMPARISONS.csv")
        spatial_means = rows(source / "SPATIAL_MEAN_SD_COMPARISON.csv")
        spatial_pairs = rows(source / "SPATIAL_PAIRED_COMPARISONS.csv")
        for dataset in dict.fromkeys(r["dataset"] for r in classification_means):
            section = text.split("## " + names[dataset] + "：", 1)[1].split("\n## ", 1)[0]
            spatial_metrics = ("pg",) if dataset == "chestxdet10" else SPATIAL if dataset == "siim_new_holdout" else ()
            for row in (r for r in classification_means if r["dataset"] == dataset):
                values = [f"{float(row[m+'_mean_percent']):.2f} ± {float(row[m+'_sd_pp']):.2f}" for m in METRICS[:2]]
                if spatial_metrics:
                    other = next(r for r in spatial_means if r["dataset"] == dataset and r["method"] == row["method"])
                    values += [f"{float(other[m+'_mean_percent']):.2f} ± {float(other[m+'_sd_pp']):.2f}" for m in spatial_metrics]
                line = "| " + " | ".join([display[row["method"]], *values]) + " |"
                assert line in section, line
                verified.append({"dataset": dataset, "row": row["method"]})
            for row in (r for r in classification_pairs if r["dataset"] == dataset and r["target"] == "query"):
                values = [f"{float(row[m+'_delta_pp']):+.2f}［{float(row[m+'_ci_low_pp']):+.2f}, {float(row[m+'_ci_high_pp']):+.2f}］" for m in METRICS[:2]]
                if row["reference"] == "readout_only":
                    primary = "| " + " | ".join([names[dataset], *values]) + " |"
                    assert primary in text and primary in insert, primary
                if spatial_metrics:
                    other = next(r for r in spatial_pairs if r["dataset"] == dataset and r["target"] == "query" and r["reference"] == row["reference"])
                    values += [f"{float(other[m+'_delta_pp']):+.2f}［{float(other[m+'_ci_low_pp']):+.2f}, {float(other[m+'_ci_high_pp']):+.2f}］" for m in spatial_metrics]
                line = "| " + " | ".join(["Q − " + display[row["reference"]], *values]) + " |"
                assert line in section, line
                verified.append({"dataset": dataset, "row": "query_minus_" + row["reference"]})
    assert len(verified) == 27
    return {"status": "passed", "verified_detail_rows": len(verified), "verified_primary_rows_in_report_and_insert": 3,
            "build_audit_sha256": sha(ROOT / "SUPPLEMENT_BUILD_AUDIT.json"), "report_sha256": sha(build["report"]),
            "insert_sha256": sha(ROOT / "MANUSCRIPT_INSERT.md")}


def main():
    catalog_path = ROOT / "protocol_siim_v1/catalog.json"
    catalog = read(catalog_path)
    identity = read(catalog["test_identity_manifest"]["path"])["rows"]
    evaluation = read(catalog["evaluation_manifest"]["path"])["rows"]
    for key in ("test_identity_manifest", "evaluation_manifest", "split_audit"):
        assert sha(catalog[key]["path"]) == catalog[key]["sha256"]
    images = [r["image"] for r in identity]
    groups = np.asarray([r["group_id"] for r in identity])
    labels = np.asarray([r["labels"][0] for r in evaluation])
    assert len(images) == 1000 and labels.sum() == 211
    assert images == [r["image"] for r in evaluation]
    assert_array_equal(groups, [r["group_id"] for r in evaluation])
    native_receipt = read(JOB / "result.json")
    for name, digest in native_receipt["artifact_sha256"].items():
        assert sha(JOB / name) == digest
    support_features, test_features = archive(JOB / "support_union_features.npz"), archive(JOB / "test_features.npz")
    assert_array_equal(test_features["image_paths"], images)
    assert_array_equal(test_features["group_ids"], groups)
    support_index = {p: i for i, p in enumerate(support_features["image_paths"].tolist())}
    plans = archive(STATS / "cluster_draws.npz")
    assert plans["counts"].shape == (2000, 1000) and int(plans["seed"]) == 20261024
    assert_array_equal(plans["group_ids"], np.unique(groups))
    assert_array_equal(plans["counts"].sum(1), np.full(2000, 1000))
    rng = np.random.Generator(np.random.PCG64(20261024))
    independent_draws = rng.integers(0, 1000, size=(2000, 1000))
    reconstructed_counts = np.stack([np.bincount(row, minlength=1000) for row in independent_draws]).astype(np.uint32)
    assert_array_equal(plans["counts"], reconstructed_counts)
    inverse = np.searchsorted(plans["group_ids"], groups)
    weights = plans["counts"][:, inverse].astype(np.float64)
    classifications, classification_draws, spatial_points, spatial_draws, seed_audits = [], [], [], [], []
    weighted_metric_max_error = 0.
    point_metric_max_error = 0.
    oof_max_error = 0.
    spatial_draw_max_error = 0.
    saved_spatial = archive(STATS / "spatial_bootstrap_metrics.npz")
    selected_replicates = (0, 7, 999, 1999)
    for seed_index, seed in enumerate(SEEDS):
        folder = JOB / f"seed_{seed}"
        native, score = archive(folder / "predictions.npz"), archive(folder / "score_predictions.npz")
        for data in (native, score):
            assert_array_equal(data["image_paths"], images)
            assert_array_equal(data["group_ids"], groups)
        scores = {m: native[m][:, 0] for m in METHODS[:4]}
        scores[METHODS[-1]] = score["logits"][:, 0]
        points = np.stack([metrics(labels, scores[m]) for m in METHODS])
        saved = archive(STATS / f"seed_{seed}/classification_metrics.npz")
        assert_array_equal(saved["method_names"], METHODS)
        assert_array_equal(saved["metric_names"], METRICS)
        assert str(saved["draw_plan_sha256"]) == str(plans["sha256"])
        point_metric_max_error = max(point_metric_max_error, float(np.max(np.abs(points - saved["point_metrics"][:, 0]))))
        assert_allclose(points, saved["point_metrics"][:, 0], atol=1e-12, rtol=0)
        for replicate in selected_replicates:
            direct = np.stack([metrics(labels, scores[m], weights[replicate]) for m in METHODS])
            target = saved["replicate_metrics"][replicate, :, 0]
            weighted_metric_max_error = max(weighted_metric_max_error, float(np.max(np.abs(direct-target))))
            assert_allclose(direct, target, atol=1e-12, rtol=0)
        classifications.append(points)
        classification_draws.append(saved["replicate_metrics"][:, :, 0])
        spatial = archive(STATS / f"seed_{seed}/spatial_per_image.npz")
        cases = spatial["cases"][:, 0]
        assert_array_equal(cases, labels)
        assert_array_equal(spatial["foreground_pixels"], [r["foreground_pixels"] for r in evaluation])
        denominator = weights @ cases
        direct_points, direct_draws = [], []
        for m in METHODS:
            values = spatial[m]
            assert values.shape == (1000, 2) and np.isfinite(values).all()
            assert np.all(values[cases == 0] == 0)
            direct_points.append(values.sum(0) / cases.sum())
            direct_draws.append((weights @ values) / denominator[:, None])
        direct_points, direct_draws = np.stack(direct_points), np.stack(direct_draws, axis=1)
        target_points = saved_spatial["point_metrics"][seed_index, :, 0]
        target_draws = saved_spatial["replicate_metrics"][seed_index, :, :, 0]
        assert_allclose(direct_points, target_points, atol=1e-12, rtol=0)
        spatial_draw_max_error = max(spatial_draw_max_error, float(np.max(np.abs(direct_draws-target_draws))))
        assert_allclose(direct_draws, target_draws, atol=1e-12, rtol=0)
        spatial_points.append(direct_points)
        spatial_draws.append(direct_draws)
        # Independent support-only OOF reconstruction. No production fitting or
        # summary helper is called in this audit.
        support = catalog["seeds"][str(seed)]["support_rows"]
        sy = np.asarray([r["labels"][0] for r in support])
        assert len(support) == 10 and sy.sum() == 5
        overlaps = {}
        for field in ("image", "group_id", "pixel_sha256", "raw_pixel_sha256"):
            overlap = set(r[field] for r in support) & set(r[field] for r in identity)
            assert not overlap, (seed, field)
            overlaps[field] = len(overlap)
        indices = [support_index[r["image"]] for r in support]
        assert_array_equal(support_features["labels"][indices, 0], sy)
        local = support_features["class_local"][indices, 0].astype(np.float64)
        margins = support_features["logits"][indices, 0].astype(np.float64)
        fit = read(folder / "score_fit.json")
        assert fit["protocol_sha256"] == sha(catalog_path) and fit["support_seed"] == seed
        assert fit["test_labels_read"] is False and fit["test_masks_read"] is False
        record = fit["methods"]["class_local_difference_score_oof"]
        assert record["test_labels_accepted"] is False and record["test_features_used_for_selection"] is False
        residual = np.zeros(10)
        held_out = np.zeros(10, dtype=int)
        assert len(fit["support_only_folds"]) == 5
        for fold in fit["support_only_folds"]:
            tr, va = np.asarray(fold["train"]), np.asarray(fold["validation"])
            assert len(tr) == 8 and len(va) == 2 and sy[tr].sum() == 4 and sy[va].sum() == 1
            assert not set(tr) & set(va) and set(tr) | set(va) == set(range(10))
            assert not set(support[i]["group_id"] for i in tr) & set(support[i]["group_id"] for i in va)
            residual[va] = local[va] @ difference_direction(local[tr], sy[tr])
            held_out[va] += 1
        assert_array_equal(held_out, np.ones(10))
        candidates = [(float(alpha), metrics(sy, margins + float(alpha)*residual)[1]) for alpha in fit["alpha_grid"]]
        for candidate, (alpha, value) in zip(record["candidates"], candidates):
            assert candidate["parameters"]["alpha"] == alpha
            oof_max_error = max(oof_max_error, abs(candidate["oof_macro_auprc"] - value))
            assert_allclose(candidate["oof_macro_auprc"], value, atol=1e-12, rtol=0)
        best_value = max(value for _, value in candidates)
        chosen = min(alpha for alpha, value in candidates if value >= best_value - 1e-12)
        assert chosen == record["selected_parameters"]["alpha"]
        predicted = test_features["logits"][:, 0].astype(np.float64) + chosen * (test_features["class_local"][:, 0].astype(np.float64) @ difference_direction(local, sy))
        assert_allclose(predicted, score["logits"][:, 0], atol=1e-12, rtol=0)
        assert_allclose(expit(predicted), score["same_feature_score"][:, 0], atol=1e-12, rtol=0)
        seed_audits.append({"seed": seed, "support_images": len(support), "positive_support": int(sy.sum()),
            "support_test_overlaps": overlaps, "OOF_folds": 5, "exactly_once_OOF_coverage": True,
            "selected_alpha_recomputed": chosen, "OOF_candidates_recomputed": len(candidates),
            "final_S_margins_recomputed": "passed"})
    cp, cd = np.stack(classifications), np.stack(classification_draws)
    sp, sd = np.stack(spatial_points), np.stack(spatial_draws)
    class_audit = confirm_summary(cp, cd, rows(STATS / "MEAN_SD_COMPARISON.csv"), rows(STATS / "PAIRED_COMPARISONS.csv"), METRICS)
    spatial_audit = confirm_summary(sp, sd, rows(STATS / "SPATIAL_MEAN_SD_COMPARISON.csv"), rows(STATS / "SPATIAL_PAIRED_COMPARISONS.csv"), SPATIAL)
    report = {"status": "passed", "created_at": datetime.now().isoformat(), "run": str(RUN),
        "audit_script_sha256": sha(__file__), "scope": "independent arithmetic, protocol, fitting-record and rendered-table audit; no model inference",
        "test_images": 1000, "positive_images": 211, "support_seeds": list(SEEDS),
        "classification_point_metrics": {"sklearn_recomputed_method_seed_count": 25, "max_abs_error": point_metric_max_error,
            "metrics": list(METRICS), "five_seed_mean_and_sample_sd_verified": True},
        "bootstrap_draw_plan": {"PCG64_seed": 20261024, "replicates": 2000, "exact_reconstruction": True,
            "same_plan_across_methods_seeds_endpoints": True},
        "classification_bootstrap_spot_check": {"sklearn_weighted_metric_replicates": list(selected_replicates),
            "method_seed_replicate_count": len(selected_replicates)*25, "max_abs_error": weighted_metric_max_error},
        "spatial_bootstrap": {"all_replicates_recomputed_from_per_image_values": True, "max_abs_error": spatial_draw_max_error,
            "foreground_pixel_counts_match_independent_preparation_audit": True,
            "limitation": "This final audit recomputes spatial aggregation; it does not rerun all per-pixel rankings or model maps."},
        "classification_summary": class_audit, "spatial_summary": spatial_audit, "OOF_candidate_max_abs_error": oof_max_error,
        "per_seed_support_and_S_fit": seed_audits, "supplement": check_supplement(),
        "limitations": ["The SIIM cohort is a new holdout from the same source, not a new-hospital external validation.",
            "DICOM identifier groups are not verified longitudinal patients; exact pixel separation does not exclude all near duplicates.",
            "Intervals condition on the five fixed support sets and test-group resampling; no support-population uncertainty or multiplicity correction.",
            "Native maps are identical for Original/Readout and Aggregation/Q by construction; retained S maps do not explain its score residual.",
            "Spatial metrics cover positive images only and use image-weighted pixel AUROC and mask PG, not Dice or normal-image false-positive segmentation."],
        "primary_query_minus_readout": {"classification": next(r for r in class_audit["comparisons"] if r["target"] == "query" and r["reference"] == "readout_only"),
            "spatial": next(r for r in spatial_audit["comparisons"] if r["target"] == "query" and r["reference"] == "readout_only")},
        "source_sha256": {str(path): sha(path) for path in (catalog_path, STATS / "RESULTS.json", STATS / "AUDIT.json",
            STATS / "MEAN_SD_COMPARISON.csv", STATS / "PAIRED_COMPARISONS.csv", STATS / "SPATIAL_PAIRED_COMPARISONS.csv",
            STATS / "spatial_bootstrap_metrics.npz", STATS / "cluster_draws.npz", JOB / "result.json", JOB / "S_FIT_RECEIPT.json")}}
    target = ROOT / "FINAL_STATISTICS_AUDIT.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "path": str(target), "primary": report["primary_query_minus_readout"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
