#!/usr/bin/env python3
"""Evaluate a frozen 2x2 query-path intervention on existing development cohorts.

Prediction archives contain no test labels. Labels and boxes are read here only
after the full five-seed prediction archives exist. All branches use the same
test-group bootstrap draws. No intervention or support seed is selected by score.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
LEGACY = ROOT.parent / "fewshot_multiseed_20261024"
STABILITY = ROOT.parent / "rsna_chest_stability_20261024"
REFERENCE_RUN = LEGACY / "runs/full_20261024_v2"
SEEDS = (42, 43, 44, 45, 46)
NATIVE = ("original", "readout_only", "aggregation_only", "query")
METHODS = NATIVE + ("score_retained_original_map",)
SCORE_KEY = "class_local_difference_score_oof"
DATASETS = ("rsna_pneumonia", "chestxdet10")
DISPLAY = {"original": "Original", "readout_only": "Readout only",
           "aggregation_only": "Aggregation only", "query": "Full Q",
           "score_retained_original_map": "S + retained Original map"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module("query_path_legacy_statistics", LEGACY / "summarize_multiseed.py")
pg_reference = load_module("query_path_pg_reference", STABILITY / "infer_multiseed.py")
from cluster_bootstrap import make_patient_draw_plan, paired_cluster_bootstrap

sha, write_json, write_csv, load_npz = legacy.sha, legacy.write_json, legacy.write_csv, legacy.load_npz


def verify_native_predictions(data, expected):
    for key in ("image_paths", "class_names", "group_ids"):
        if key not in data or not np.array_equal(data[key], expected[key]):
            raise ValueError(f"Prediction/protocol identity mismatch: {key}")
    if "labels" in data:
        raise ValueError("Inference archives must not contain evaluation labels")
    for branch in NATIVE:
        value = data.get(branch)
        if value is None or value.shape != expected["labels"].shape:
            raise ValueError(f"Invalid prediction shape: {branch}")
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError(f"Native scores must be finite sigmoid probabilities: {branch}")


def support_signature(spec, seed):
    return [(r["image"], r["labels"], str(r["group_id"]))
            for r in spec["seeds"][str(seed)]["support_rows"]]


def load_score_reference(dataset, seed, expected, spec, new_predictions):
    folder = REFERENCE_RUN / f"AlphaRAD_{dataset}/evaluation/seed_{seed}"
    result_path, prediction_path = folder / "result.json", folder / "predictions.npz"
    record = json.loads(result_path.read_text())
    old_catalog_path = LEGACY / "protocol_v1/catalog.json"
    old_catalog = json.loads(old_catalog_path.read_text())
    old_entry = old_catalog["datasets"][dataset]
    protocol_path = LEGACY / f"protocol_v1/{dataset}.json"
    old_spec = legacy.frozen_json(old_entry)
    seed_protocol_path = Path(old_entry["seeds"][str(seed)]["path"])
    if sha(seed_protocol_path) != record["protocol_sha256"] or sha(seed_protocol_path) != old_entry["seeds"][str(seed)]["sha256"]:
        raise ValueError("Archived comparator protocol hash changed")
    if sha(old_catalog_path) != record["catalog_sha256"]:
        raise ValueError("Archived comparator catalog hash changed")
    if sha(prediction_path) != record["predictions_sha256"]:
        raise ValueError("Archived comparator predictions hash changed")
    if record["support_seed"] != seed or support_signature(spec, seed) != support_signature(old_spec, seed):
        raise ValueError("Archived S and new native interventions use different support data")
    old = load_npz(prediction_path)
    legacy.verify_predictions(old, expected, methods=("original", "query", SCORE_KEY))
    errors = {}
    for branch in ("original", "query"):
        errors[branch] = float(np.max(np.abs(old[branch] - new_predictions[branch])))
        np.testing.assert_allclose(new_predictions[branch], old[branch], rtol=2e-5, atol=2e-5,
                                   err_msg=f"Same-support native endpoint regression failed: {dataset}/{seed}/{branch}")
    provenance = {"result_path": str(result_path), "result_sha256": sha(result_path),
                  "archived_predictions_path": str(prediction_path), "archived_predictions_sha256": sha(prediction_path),
                  "protocol_path": str(protocol_path), "protocol_sha256": sha(protocol_path),
                  "seed_protocol_path": str(seed_protocol_path), "seed_protocol_sha256": sha(seed_protocol_path),
                  "support_identity_verified": True, "native_endpoint_max_abs_error": errors,
                  "S_score_key": SCORE_KEY,
                  "S_spatial_role": "retained Original map by construction; does not explain S score residual"}
    return old[SCORE_KEY], provenance


def select_spatial_maps(data, expected, rows):
    paths = data["image_paths"].tolist()
    if len(paths) != len(set(paths)) or not set(paths) <= set(expected["image_paths"].tolist()):
        raise ValueError("Spatial archive image identities are duplicated or outside the test cohort")
    lookup = {path: i for i, path in enumerate(paths)}
    selected = [row["image"] for row in rows]
    if not set(selected) <= set(paths):
        raise ValueError("Spatial archive lacks part of the frozen spatial test cohort")
    if not np.array_equal(data["class_names"], expected["class_names"]):
        raise ValueError("Spatial archive class order differs")
    test_lookup = {path: i for i, path in enumerate(expected["image_paths"].tolist())}
    if not np.array_equal(data["group_ids"], expected["group_ids"][[test_lookup[p] for p in paths]]):
        raise ValueError("Spatial archive group identities differ")
    if "includes_cls" in data and not bool(data["includes_cls"]):
        raise ValueError("This AlphaRAD experiment requires the unmodified native CLS+patch map")
    if "grid_size" in data and not np.all(np.asarray(data["grid_size"]) == 37):
        raise ValueError("Unexpected native patch grid")
    result = {"image_paths": np.asarray(selected), "class_names": expected["class_names"],
              "group_ids": expected["group_ids"][[test_lookup[p] for p in selected]]}
    for branch in NATIVE:
        array = data[branch]
        if array.shape != (len(paths), len(expected["class_names"]), 1370) or not np.isfinite(array).all():
            raise ValueError(f"Invalid native spatial map: {branch}")
        result[branch] = array[[lookup[p] for p in selected]]
    # Both equalities follow from the intervention definition: grounding uses
    # the attention query only, before context aggregation and readout.
    equality = {}
    for a, b in (("original", "readout_only"), ("aggregation_only", "query")):
        equality[a + "__equals__" + b] = float(np.max(np.abs(result[a] - result[b])))
        np.testing.assert_array_equal(result[a], result[b], err_msg="Readout unexpectedly changed the native attention map")
    return result, equality


def bbox_softmax_energy(maps, rows, classes):
    """Threshold-free softmax mass inside the union of ground-truth boxes.

    Interpolate native mean-head scaled cosine logits exactly as native PG:
    square image -> full original resolution; nonsquare -> 518x518 native crop.
    Softmax across pixels within this visible map, then sum mass in the box
    union. Non-square pixels use the same inverse crop and rounding as native
    PG. No temperature, threshold, box, or seed is selected using test outcomes.
    This diagnostic is affected by native logit scale and lesion area; it is
    neither a segmentation metric nor a calibrated attention probability.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F
    patches = torch.as_tensor(maps[..., 1:], dtype=torch.float32).reshape(len(rows), len(classes), 37, 37)
    values = np.zeros((len(rows), len(classes)), dtype=np.float64)
    cases = np.zeros_like(values, dtype=np.uint8)
    for i, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            width, height = image.size
        if height == width:
            target = (height, width)
            ys, xs = np.arange(height), np.arange(width)
        else:
            scale = 518 / float(min(height, width))
            resized_h, resized_w = int(round(height * scale)), int(round(width * scale))
            top, left = max((resized_h - 518) // 2, 0), max((resized_w - 518) // 2, 0)
            target = (518, 518)
            ys = np.rint((np.arange(518) + top) / scale).astype(int)
            xs = np.rint((np.arange(518) + left) / scale).astype(int)
        for j, cls in enumerate(classes):
            boxes = [box for box, name in zip(row["boxes"], row["class_names_present"]) if name == cls]
            if not boxes:
                continue
            heat = F.interpolate(patches[i, j][None, None], size=target, mode="bilinear", align_corners=False)[0, 0]
            mass = torch.softmax(heat.flatten(), dim=0).reshape(target).numpy()
            mask = np.zeros(target, dtype=bool)
            for x1, y1, x2, y2 in boxes:
                mask |= ((ys[:, None] >= y1) & (ys[:, None] <= y2) &
                         (xs[None, :] >= x1) & (xs[None, :] <= x2))
            values[i, j] = mass[mask].sum(dtype=np.float64)
            cases[i, j] = 1
    return values, cases


def spatial_group_bootstrap(values, cases, group_ids, plan):
    """Class-specific case means, then class macro; keep empty-class NaN."""
    if values.shape != cases.shape or np.any(cases.sum(0) == 0):
        raise ValueError("Invalid spatial sufficient statistics or empty fixed class")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > cases + 1e-6):
        raise ValueError("Spatial values must be finite case-weighted fractions")
    indices = np.searchsorted(plan.patient_ids, group_ids)
    if np.any(indices >= len(plan.patient_ids)) or not np.array_equal(plan.patient_ids[indices], group_ids):
        raise ValueError("Unknown spatial group")
    weights = plan.counts[:, indices].astype(np.float64)
    denominator = weights @ cases
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = weights @ values / denominator
    draws[denominator == 0] = np.nan
    return values.sum(0) / cases.sum(0), draws


def native_pg_hits(maps, rows, classes):
    """Native AlphaRAD PG geometry; PIL reads dimensions without OpenCV dependency.

    Interpolation, row-major argmax and crop inverse rounding are unchanged from
    rsna_chest_stability/infer_multiseed.py:spatial_hits. All Original/Q hits are
    independently checked against the archived native evaluator below.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F
    patches = torch.as_tensor(maps[..., 1:], dtype=torch.float32).reshape(len(rows), len(classes), 37, 37)
    hits = np.zeros((len(rows), len(classes)), dtype=np.uint8)
    cases = np.zeros_like(hits)
    for i, row in enumerate(rows):
        with Image.open(row["image"]) as image:
            width, height = image.size
        for j, cls in enumerate(classes):
            boxes = [box for box, name in zip(row["boxes"], row["class_names_present"]) if name == cls]
            if not boxes:
                continue
            if height == width:
                heat = F.interpolate(patches[i, j][None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
                index = int(torch.argmax(heat).item())
                y_peak, x_peak = index // width, index % width
            else:
                scale = 518 / float(min(height, width))
                resized_h, resized_w = int(round(height * scale)), int(round(width * scale))
                top, left = max((resized_h - 518) // 2, 0), max((resized_w - 518) // 2, 0)
                heat = F.interpolate(patches[i, j][None, None], size=(518, 518), mode="bilinear", align_corners=False)[0, 0]
                index = int(torch.argmax(heat).item())
                y_peak = int(round((index // 518 + top) / scale))
                x_peak = int(round((index % 518 + left) / scale))
            cases[i, j] = 1
            hits[i, j] = int(any(x1 <= x_peak <= x2 and y1 <= y_peak <= y2 for x1, y1, x2, y2 in boxes))
    per_class = {cls: float(hits[:, j].sum() / cases[:, j].sum()) for j, cls in enumerate(classes)}
    return hits, cases, {"pointing_game": float(np.mean(list(per_class.values()))),
                         "per_class": per_class, "cases": cases.sum(0).tolist()}


def evaluate_spatial(data, rows, expected, plan, dataset, seed, folder):
    selected, equality = select_spatial_maps(data, expected, rows)
    values, native = {}, {}
    for branch in ("original", "query"):
        hits, cases, metric = native_pg_hits(selected[branch], rows, expected["class_names"].tolist())
        energy, energy_cases = bbox_softmax_energy(selected[branch], rows, expected["class_names"].tolist())
        np.testing.assert_array_equal(cases, energy_cases)
        values[branch] = (hits, energy, cases)
        native[branch] = metric
    for branch, source in (("readout_only", "original"), ("aggregation_only", "query"),
                           (METHODS[-1], "original")):
        values[branch] = values[source]
    old_path = REFERENCE_RUN / f"AlphaRAD_{dataset}/evaluation/seed_{seed}/spatial.npz"
    old_result = json.loads((old_path.parent / "result.json").read_text())
    if sha(old_path) != old_result["spatial_sha256"]:
        raise ValueError("Archived native spatial sufficient statistics hash changed")
    old = load_npz(old_path)
    old_lookup = {p: i for i, p in enumerate(old["image_paths"].tolist())}
    indices = [old_lookup[p] for p in selected["image_paths"].tolist()]
    if not np.array_equal(old["class_names"], expected["class_names"]):
        raise ValueError("Archived spatial class order differs")
    for branch, old_branch in (("original", "original"), ("query", "query")):
        np.testing.assert_array_equal(values[branch][0], old[old_branch + "_hits"][indices],
                                      err_msg=f"PG endpoint regression failed: {seed}/{branch}")
        np.testing.assert_array_equal(values[branch][2], old[old_branch + "_cases"][indices])
    points, draws, sufficient = [], [], {}
    for branch in METHODS:
        hit, energy, cases = values[branch]
        metric_points, metric_draws = [], []
        for value in (hit, energy):
            point, draw = spatial_group_bootstrap(value, cases, selected["group_ids"], plan)
            metric_points.append(point)
            metric_draws.append(draw)
        points.append(np.stack(metric_points, axis=-1))
        draws.append(np.stack(metric_draws, axis=-1))
        sufficient.update({branch + "_hits": hit, branch + "_energy": energy, branch + "_cases": cases})
    np.savez_compressed(folder / "spatial_statistics.npz", **{k: selected[k] for k in ("image_paths", "class_names", "group_ids")}, **sufficient)
    audit = {"map_equalities_max_abs_error": equality, "native_pg": native,
             "archived_spatial": {"path": str(old_path), "sha256": sha(old_path)},
             "native_endpoint_hits_cases_regression": "passed", "spatial_images": len(rows),
             "S_map": "retained Original by definition; S adapted map remains unavailable",
             "interpretation": "Q and aggregation-only maps are mathematically equal; spatial equality is not evidence for nonlinear reaggregation"}
    return np.stack(points), np.stack(draws, axis=1), audit


def interaction_summary(dataset, points, draws, metric_names, unit):
    """Difference of metric changes, not an interaction of image probabilities."""
    coefficients = np.asarray([1., -1., -1., 1., 0.])
    macro_point, macro_draw = points.mean(axis=2), draws.mean(axis=3)
    values = np.einsum("sbm,b->sm", macro_point, coefficients)
    samples = np.einsum("srbm,b->srm", macro_draw, coefficients)
    row = {"model": "AlphaRAD", "dataset": dataset, "resampling_unit": unit,
           "contrast": "(Q - aggregation_only) - (readout_only - Original)",
           "interpretation": "factorial interaction on the reported metric scale"}
    for i, metric in enumerate(metric_names):
        ci = legacy.percent_interval(values[:, i].mean(), samples[:, :, i].mean(0))
        row.update({metric + "_delta_pp": float(values[:, i].mean() * 100),
                    metric + "_gain_sd_pp": float(values[:, i].std(ddof=1) * 100),
                    **legacy.interval_columns(metric, ci)})
    return row


def process_classification(data, expected, spec, plan, folder, provenance):
    folder.mkdir(parents=True, exist_ok=True)
    result = paired_cluster_bootstrap(expected["labels"], {b: data[b] for b in METHODS},
        expected["group_ids"], task_names=spec["class_names"], plan=plan)
    audit_error = legacy.metric_audit(expected["labels"], data, result.point_metrics, methods=METHODS)
    np.savez_compressed(folder / "classification_metrics.npz", point_metrics=result.point_metrics,
        replicate_metrics=result.replicate_metrics, method_names=np.asarray(METHODS),
        class_names=expected["class_names"], metric_names=np.asarray(legacy.METRIC_NAMES),
        draw_plan_sha256=np.asarray(plan.sha256))
    write_json(folder / "provenance.json", {**provenance, "sklearn_metric_max_abs_error": audit_error,
               "draw_plan_sha256": plan.sha256})
    return result


def dataset_cache_identity(args, dataset, source_sha256, protocol_entry):
    """Bind resumable statistics to every prediction and comparator artifact."""
    artifacts = {}
    for seed in SEEDS:
        current = args.run / f"AlphaRAD_{dataset}/seed_{seed}"
        old = REFERENCE_RUN / f"AlphaRAD_{dataset}/evaluation/seed_{seed}"
        paths = [current / "predictions.npz", old / "result.json", old / "predictions.npz"]
        if dataset == "chestxdet10":
            paths += [current / "native_maps.npz", old / "spatial.npz"]
        for path in paths:
            artifacts[str(path)] = sha(path)
    old_protocol = LEGACY / f"protocol_v1/{dataset}.json"
    artifacts[str(old_protocol)] = sha(old_protocol)
    return {"source_sha256": source_sha256, "artifacts": artifacts,
            "protocol": protocol_entry, "replicates": args.replicates, "seed": args.seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--protocol", type=Path, default=STABILITY / "protocol_v1/catalog.json")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    # All requested inference must have finished before test labels are opened.
    for dataset in args.datasets:
        for seed in SEEDS:
            folder = args.run / f"AlphaRAD_{dataset}/seed_{seed}"
            required = [folder / "predictions.npz"]
            if dataset == "chestxdet10":
                required.append(folder / "native_maps.npz")
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(f"All five-seed predictions must exist before evaluation: {path}")
    catalog = json.loads(args.protocol.read_text())
    output = args.run / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "running", "created_at": datetime.now().isoformat(),
        "experiment": "single-view frozen native-query 2x2 intervention", "model": "AlphaRAD",
        "support_seeds": list(SEEDS), "methods": list(METHODS), "datasets": {},
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "primary_comparison": "query - readout_only", "primary_classification_metric": "auroc",
        "uncertainty_note": legacy.UNCERTAINTY_NOTE,
        "scope": "existing development cohorts; not independent external validation",
        "source_sha256": {str(Path(__file__)): sha(__file__), str(args.protocol): sha(args.protocol),
            str(LEGACY / "summarize_multiseed.py"): sha(LEGACY / "summarize_multiseed.py"),
            str(STABILITY / "infer_multiseed.py"): sha(STABILITY / "infer_multiseed.py"), str(legacy.ENGINE): sha(legacy.ENGINE)},
        "spatial_definition": "native PG and diagnostic box-union softmax energy, class-macro over annotated positive cases; same maps for Original/Readout/S and Aggregation/Q by construction"}
    pooled = {key: [] for key in ("per_seed", "means", "per_class", "comparisons", "seed_comparisons", "class_comparisons")}
    spatial_pooled = {key: [] for key in pooled}
    interactions, spatial_interactions = [], []
    for dataset in args.datasets:
        spec = legacy.frozen_json(catalog["datasets"][dataset])
        if set(spec["seeds"]) != {str(s) for s in SEEDS}:
            raise ValueError("The declared five support seeds changed")
        expected = legacy.prepare_expected(spec)
        cache_identity = dataset_cache_identity(args, dataset, summary["source_sha256"], catalog["datasets"][dataset])
        complete_path = output / dataset / "COMPLETE.json"
        if complete_path.exists():
            cached = json.loads(complete_path.read_text())
            if cached["identity"] != cache_identity:
                raise ValueError(f"Completed evaluation provenance changed; choose a fresh run or explicitly archive old evaluation: {complete_path}")
            print(f"REUSE verified completed evaluation: {dataset}", flush=True)
            for key in pooled:
                pooled[key].extend(cached["classification"][key])
                spatial_pooled[key].extend(cached["spatial"][key])
            interactions.append(cached["interaction"])
            if cached["spatial_interaction"] is not None:
                spatial_interactions.append(cached["spatial_interaction"])
            summary["datasets"][dataset] = cached["info"]
            continue
        plan = make_patient_draw_plan(expected["group_ids"], n_resamples=args.replicates, seed=args.seed)
        legacy.save_group_plan(plan, output / f"{dataset}_cluster_draws.npz", spec["resampling_unit"])
        spatial_rows = legacy.frozen_json(spec["spatial_manifest"])["rows"] if dataset == "chestxdet10" else None
        points, draws, spatial_points, spatial_draws, audits = [], [], [], [], []
        original_reference = None
        for seed in SEEDS:
            print(f"EVALUATE {dataset} support seed {seed}", flush=True)
            folder = args.run / f"AlphaRAD_{dataset}/seed_{seed}"
            prediction_path = folder / "predictions.npz"
            data = load_npz(prediction_path)
            verify_native_predictions(data, expected)
            if original_reference is None:
                original_reference = data["original"]
            np.testing.assert_array_equal(data["original"], original_reference, err_msg="Original changed across support seeds")
            score, audit = load_score_reference(dataset, seed, expected, spec, data)
            data[METHODS[-1]] = score
            audit.update(predictions_path=str(prediction_path), predictions_sha256=sha(prediction_path),
                         support_seed=seed, dataset_protocol_sha256=catalog["datasets"][dataset]["sha256"])
            stats_folder = output / dataset / f"seed_{seed}"
            result = process_classification(data, expected, spec, plan, stats_folder, audit)
            points.append(result.point_metrics)
            draws.append(result.replicate_metrics)
            if spatial_rows is not None:
                maps_path = folder / "native_maps.npz"
                p, d, spatial_audit = evaluate_spatial(load_npz(maps_path), spatial_rows, expected, plan, dataset, seed, stats_folder)
                spatial_audit.update(native_maps_path=str(maps_path), native_maps_sha256=sha(maps_path))
                write_json(stats_folder / "spatial_audit.json", spatial_audit)
                spatial_points.append(p)
                spatial_draws.append(d)
                audit["spatial"] = spatial_audit
            audits.append(audit)
        p, d = np.stack(points), np.stack(draws)
        aggregates = legacy.summarize_arrays("AlphaRAD", dataset, p, d, spec["class_names"], spec["resampling_unit"], methods=METHODS)
        for key in pooled:
            pooled[key].extend(aggregates[key])
        interaction = interaction_summary(dataset, p, d, legacy.METRIC_NAMES, spec["resampling_unit"])
        interactions.append(interaction)
        info = {"test_images": len(expected["labels"]), "test_groups": len(plan.patient_ids),
                "resampling_unit": spec["resampling_unit"], "classification": aggregates["detail"], "source_audits": audits}
        spatial_aggregates = {key: [] for key in pooled}
        spatial_interaction = None
        if spatial_points:
            p, d = np.stack(spatial_points), np.stack(spatial_draws)
            s = legacy.summarize_arrays("AlphaRAD", dataset, p, d, spec["class_names"], spec["resampling_unit"], methods=METHODS,
                                        metric_names=("pg", "bbox_softmax_energy"))
            for key in spatial_pooled:
                spatial_pooled[key].extend(s[key])
                spatial_aggregates[key] = s[key]
            spatial_interaction = interaction_summary(dataset, p, d, ("pg", "bbox_softmax_energy"), spec["resampling_unit"])
            spatial_interactions.append(spatial_interaction)
            info["spatial"] = s["detail"]
            np.savez_compressed(output / f"{dataset}_spatial_metrics.npz", point_metrics=p, replicate_metrics=d,
                method_names=np.asarray(METHODS), metric_names=np.asarray(["pg", "bbox_softmax_energy"]),
                draw_plan_sha256=np.asarray(plan.sha256))
        summary["datasets"][dataset] = info
        write_json(complete_path, {"identity": cache_identity, "info": info,
                   "classification": {key: aggregates[key] for key in pooled},
                   "spatial": spatial_aggregates, "interaction": interaction,
                   "spatial_interaction": spatial_interaction})
    filenames = {"per_seed": "PER_SEED_RESULTS.csv", "means": "MEAN_SD_COMPARISON.csv", "per_class": "PER_CLASS_MEAN_SD.csv",
                 "comparisons": "PAIRED_COMPARISONS.csv", "seed_comparisons": "PER_SEED_PAIRED_COMPARISONS.csv",
                 "class_comparisons": "PER_CLASS_PAIRED_COMPARISONS.csv"}
    for key, name in filenames.items():
        write_csv(output / name, pooled[key])
        write_csv(output / ("SPATIAL_" + name), spatial_pooled[key])
    write_csv(output / "FACTORIAL_INTERACTION.csv", interactions)
    write_csv(output / "SPATIAL_FACTORIAL_INTERACTION.csv", spatial_interactions)
    lines = ["AlphaRAD native-query path intervention: five support seeds, single view", "",
             "Existing development tests; this is not independent external validation.",
             "Mean±sample SD across seeds 42–46. CI: shared paired test-group bootstrap conditional on those supports.",
             "RSNA groups are patients; ChestXDet10 groups are images/exact duplicates, not verified patients.",
             "S reuses the archived same-support score residual; its retained Original map does not explain the new score.",
             "Native grounding depends only on the attention query: Q=Aggregation maps and Readout=Original maps by construction.", ""]
    spatial_mean_lookup = {(r["dataset"], r["method"]): r for r in spatial_pooled["means"]}
    spatial_pair_lookup = {(r["dataset"], r["target"], r["reference"]): r for r in spatial_pooled["comparisons"]}
    for dataset in args.datasets:
        lines += [dataset, "", "| Method / contrast | AUROC | AUPRC | PG |", "|---|---:|---:|---:|"]
        for row in (r for r in pooled["means"] if r["dataset"] == dataset):
            values = [f"{row[m + '_mean_percent']:.2f} ± {row[m + '_sd_pp']:.2f}" for m in ("auroc", "auprc")]
            sr = spatial_mean_lookup.get((dataset, row["method"]))
            values.append("NA" if sr is None else f"{sr['pg_mean_percent']:.2f} ± {sr['pg_sd_pp']:.2f}")
            lines.append("| " + " | ".join([DISPLAY[row["method"]], *values]) + " |")
        for row in (r for r in pooled["comparisons"] if r["dataset"] == dataset and r["target"] == "query"):
            values = [legacy.ci_text(row, m) for m in ("auroc", "auprc")]
            sr = spatial_pair_lookup.get((dataset, row["target"], row["reference"]))
            values.append("NA" if sr is None else legacy.ci_text(sr, "pg"))
            lines.append("| " + " | ".join(["Q − " + DISPLAY[row["reference"]], *values]) + " |")
        lines += [""]
    lines += ["Absolute values are percent; contrasts/intervals are percentage points. All predeclared contrasts are in CSV.",
              "Factorial interaction is (Q−Aggregation)−(Readout−Original) on the metric scale, not probability-scale causality.",
              "Box softmax energy: bilinearly interpolate native mean-head scaled cosine logits using PG's geometry, spatial softmax, then mass inside the union of true boxes. Positive cases per class, then macro average. No thresholds; diagnostic, not segmentation or calibrated attribution.",
              "This study changes attention/aggregation and final readout separately. It does not establish necessity for all models or independent-cohort generalization."]
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    (output / "PAPER_TABLE.md").write_text("\n".join(lines) + "\n")
    summary["status"] = "complete"
    summary["coverage"] = {"classification_seed_rows": len(pooled["per_seed"]), "classification_means": len(pooled["means"]),
                           "spatial_seed_rows": len(spatial_pooled["per_seed"]), "spatial_means": len(spatial_pooled["means"])}
    write_json(output / "RESULTS.json", summary)
    artifacts = [p for p in output.iterdir() if p.is_file() and p.name != "AUDIT.json"]
    write_json(output / "AUDIT.json", {"status": "passed", "source_sha256": summary["source_sha256"],
               "artifact_sha256": {str(p): sha(p) for p in artifacts}, "coverage": summary["coverage"]})
    print(json.dumps({"status": "complete", "output": str(output), "coverage": summary["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
