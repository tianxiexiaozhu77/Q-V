#!/usr/bin/env python3
"""Evaluate the frozen same-source SIIM holdout without spatial threshold fitting.

The cohort was withheld from the preceding local development evaluations; its
DICOM grouping is not verified patient identity or independent-hospital status.
All labels/masks are opened after five-seed inference and support-only S fitting.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import evaluate as base

SEEDS, NATIVE, METHODS = base.SEEDS, base.NATIVE, base.METHODS
DATASET = "siim_new_holdout"
MASK_FORMAT = "siim_relative_gap_zero_based_column_major_union"
UNIT = "dicom_identifier_group"


def frozen_json(entry, parent):
    if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
        raise ValueError("Frozen manifest needs path and sha256")
    path = Path(entry["path"])
    if not path.is_absolute():
        path = parent / path
    if base.sha(path) != entry["sha256"]:
        raise ValueError(f"Frozen manifest hash mismatch: {path}")
    return json.loads(path.read_text()), path


def decode_siim_mask(rles, shape):
    """Author SIIM encoding: relative zero-based gaps in column-major order."""
    height, width = map(int, shape)
    if height <= 0 or width <= 0:
        raise ValueError("Invalid mask shape")
    flat = np.zeros(height * width, dtype=bool)
    for rle in rles:
        rle = str(rle).strip()
        if rle in ("-1", ""):
            continue
        parts = np.asarray([int(token) for token in rle.split()], dtype=np.int64)
        if len(parts) % 2 or np.any(parts < 0):
            raise ValueError("Invalid SIIM relative-gap RLE")
        position = 0
        for gap, length in parts.reshape(-1, 2):
            position += int(gap)
            end = position + int(length)
            if end > len(flat):
                raise ValueError("SIIM RLE extends outside image")
            flat[position:end] = True
            position = end
    return flat.reshape(width, height).T


def pixel_metrics(patches, mask):
    """Within-positive-image pixel ranking and one row-major peak hit."""
    import torch
    import torch.nn.functional as F
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError("Positive-image pixel AUROC requires both foreground and background")
    heat = F.interpolate(torch.as_tensor(patches, dtype=torch.float32)[None, None],
        size=mask.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    score = float(roc_auc_score(mask.ravel(), heat.ravel()))
    peak = int(np.argmax(heat))
    return score, float(mask.flat[peak])


def spatial_worker(item):
    index, patches, row = item
    mask = decode_siim_mask(row["rle"], row["original_shape"])
    auroc, pg = pixel_metrics(patches, mask)
    return index, auroc, pg, int(mask.sum())


def initialize_worker():
    import torch
    torch.set_num_threads(1)


def spatial_image_values(maps, rows, workers):
    labels = np.asarray([r["labels"][0] for r in rows])
    values = np.zeros((len(rows), 2), dtype=np.float64)
    cases = labels[:, None].astype(np.uint8)
    items = [(i, maps[i, 0, 1:].reshape(37, 37), row) for i, row in enumerate(rows) if labels[i] == 1]
    foreground = np.zeros(len(rows), dtype=np.int64)
    if workers == 1:
        initialize_worker()
        results = map(spatial_worker, items)
        for i, auc, pg, pixels in results:
            values[i] = (auc, pg)
            foreground[i] = pixels
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
            for i, auc, pg, pixels in pool.map(spatial_worker, items, chunksize=2):
                values[i] = (auc, pg)
                foreground[i] = pixels
    return values, cases, foreground


def validate_manifest(catalog, identity, evaluation):
    if catalog["class_names"] != ["Pneumothorax"]:
        raise ValueError("Unexpected SIIM class")
    if evaluation.get("mask_format") != MASK_FORMAT:
        raise ValueError("Unsupported or unspecified SIIM mask encoding")
    ids, rows = identity["rows"], evaluation["rows"]
    if len(ids) != len(rows) or len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Invalid evaluation image count or identities")
    for identifier, row in zip(ids, rows):
        if identifier["image"] != row["image"] or str(identifier["group_id"]) != str(row["group_id"]):
            raise ValueError("Evaluation and label-free inference manifest order/identity differ")
        if row["labels"] not in ([0], [1]) or row["canonical_shape"] != [512, 512]:
            raise ValueError("Unexpected label or native input geometry")
        if row["original_shape"] != [1024, 1024]:
            raise ValueError("This protocol requires verified original 1024-square masks")
        mask = decode_siim_mask(row["rle"], row["original_shape"])
        if int(mask.any()) != row["labels"][0] or mask.all():
            raise ValueError("Image label and available SIIM mask disagree")
    expected = {"image_paths": np.asarray([r["image"] for r in rows]),
                "group_ids": np.asarray([str(r["group_id"]) for r in rows]),
                "class_names": np.asarray(catalog["class_names"]),
                "labels": np.asarray([r["labels"] for r in rows], dtype=np.uint8)}
    if not 0 < expected["labels"].sum() < len(rows):
        raise ValueError("Unevaluable classification cohort")
    for seed in SEEDS:
        support = catalog["seeds"][str(seed)]["support_rows"]
        if set(str(r["group_id"]) for r in support) & set(expected["group_ids"].tolist()):
            raise ValueError("Support/test DICOM identifier groups overlap")
        if set(r["image"] for r in support) & set(expected["image_paths"].tolist()):
            raise ValueError("Support/test images overlap")
    return expected


def native_maps(data, expected):
    for key in ("image_paths", "class_names", "group_ids"):
        if not np.array_equal(data[key], expected[key]):
            raise ValueError(f"Native spatial identity mismatch: {key}")
    if not bool(data["includes_cls"]) or int(data["grid_size"]) != 37:
        raise ValueError("Unexpected native CLS or patch-grid metadata")
    for branch in NATIVE:
        if data[branch].shape != (len(expected["labels"]), 1, 1370) or not np.isfinite(data[branch]).all():
            raise ValueError("Expected finite native CLS + 37-square map")
    for a, b in (("original", "readout_only"), ("query", "aggregation_only")):
        np.testing.assert_array_equal(data[a], data[b], err_msg="Unexpected readout effect on native map")
    return data


def verify_inference_receipt(receipt, job, protocol):
    if receipt["protocol_sha256"] != base.sha(protocol):
        raise ValueError("Inference protocol differs from evaluation protocol")
    if any(receipt[key] for key in ("test_labels_read", "test_masks_read", "test_rle_read")):
        raise ValueError("Inference must not access test annotations")
    if not receipt["frozen_model"] or receipt["support_seeds"] != list(SEEDS) or receipt["views"] != 1 or receipt["eta"] != .3:
        raise ValueError("Inference does not match the frozen intervention protocol")
    sources = {}
    for name, digest in receipt["artifact_sha256"].items():
        path = job / name
        if base.sha(path) != digest:
            raise ValueError(f"Frozen native inference artifact hash mismatch: {path}")
        sources[str(path)] = digest
    for seed in SEEDS:
        for name in ("predictions.npz", "native_maps.npz", "query_fit.npz", "support_manifest.json"):
            relative = f"seed_{seed}/{name}"
            if receipt["seed_artifacts"][str(seed)][name] != receipt["artifact_sha256"][relative]:
                raise ValueError("Native seed receipt and inference receipt disagree")
    return sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.workers > 8:
        p.error("Use between one and eight CPU workers")
    args.run, args.protocol = args.run.resolve(), args.protocol.resolve()
    job = args.run / "AlphaRAD_siim_new_holdout"
    # Receipts and all prediction files are required before the evaluation
    # manifest containing test labels/masks is opened.
    infer_receipt = json.loads((job / "result.json").read_text())
    fit_receipt = json.loads((job / "S_FIT_RECEIPT.json").read_text())
    if infer_receipt["status"] != "complete" or fit_receipt["status"] != "complete":
        raise ValueError("Full inference and all five support-only S fits must finish first")
    if fit_receipt["test_labels_read"] or fit_receipt["test_masks_read"]:
        raise ValueError("S fitting must not access test labels/masks")
    sources = {str(path): base.sha(path) for path in (Path(__file__), ROOT / "evaluate.py",
        args.protocol, job / "result.json", job / "S_FIT_RECEIPT.json", base.legacy.ENGINE,
        base.LEGACY / "summarize_multiseed.py")}
    sources.update(verify_inference_receipt(infer_receipt, job, args.protocol))
    for seed in SEEDS:
        for name in ("predictions.npz", "native_maps.npz", "score_predictions.npz", "score_fit.json"):
            path = job / f"seed_{seed}" / name
            sources[str(path)] = base.sha(path)
            if name in ("score_predictions.npz", "score_fit.json") and fit_receipt["artifacts"][str(seed)][name] != sources[str(path)]:
                raise ValueError("Frozen S fit artifact hash mismatch")
    catalog = json.loads(args.protocol.read_text())
    identity, identity_path = frozen_json(catalog["test_identity_manifest"], args.protocol.parent)
    evaluation, evaluation_path = frozen_json(catalog["evaluation_manifest"], args.protocol.parent)
    sources.update({str(identity_path): base.sha(identity_path), str(evaluation_path): base.sha(evaluation_path)})
    expected = validate_manifest(catalog, identity, evaluation)
    if infer_receipt["test_count"] != len(expected["labels"]) or infer_receipt["full_test_count"] != len(expected["labels"]):
        raise ValueError("Incomplete native test cohort")
    rows = evaluation["rows"]
    out = args.run / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RESULTS.json").exists():
        existing = json.loads((out / "RESULTS.json").read_text())
        if existing.get("source_sha256") != sources or existing.get("bootstrap_replicates") != args.replicates or existing.get("bootstrap_seed") != args.seed:
            raise ValueError("Evaluation already exists with changed provenance; use a new output")
        if existing.get("status") == "complete":
            print("Verified evaluation already complete", flush=True)
            return
    plan = base.make_patient_draw_plan(expected["group_ids"], n_resamples=args.replicates, seed=args.seed)
    base.legacy.save_group_plan(plan, out / "cluster_draws.npz", UNIT)
    class_points, class_draws, space_points, space_draws, audits = [], [], [], [], []
    original_predictions, original_map, original_spatial = None, None, None
    for seed in SEEDS:
        print(f"EVALUATE SIIM support seed {seed}", flush=True)
        folder = job / f"seed_{seed}"
        data = base.load_npz(folder / "predictions.npz")
        base.verify_native_predictions(data, expected)
        score = base.load_npz(folder / "score_predictions.npz")
        score_fit = json.loads((folder / "score_fit.json").read_text())
        if score_fit["protocol_sha256"] != base.sha(args.protocol) or score_fit["support_seed"] != seed:
            raise ValueError("S fit protocol/seed mismatch")
        for key in ("image_paths", "class_names", "group_ids"):
            np.testing.assert_array_equal(data[key], score[key])
        # Ranking uses the original uncompressed S margin to avoid artificial
        # sigmoid ties, matching the previous same-feature score comparator.
        if score["logits"].shape != expected["labels"].shape or not np.isfinite(score["logits"]).all():
            raise ValueError("Invalid S margins")
        data[METHODS[-1]] = score["logits"]
        maps = native_maps(base.load_npz(folder / "native_maps.npz"), expected)
        if original_predictions is None:
            original_predictions, original_map = data["original"], maps["original"]
        np.testing.assert_array_equal(data["original"], original_predictions)
        np.testing.assert_array_equal(maps["original"], original_map)
        stats = out / f"seed_{seed}"
        stats.mkdir(exist_ok=True)
        classification = base.process_classification(data, expected,
            {"class_names": catalog["class_names"]}, plan, stats,
            {"support_seed": seed, "resampling_unit": UNIT, "source_sha256": sources})
        class_points.append(classification.point_metrics)
        class_draws.append(classification.replicate_metrics)
        print(f"SPATIAL SIIM support seed {seed}: {int(expected['labels'].sum())} positive images", flush=True)
        if original_spatial is None:
            original_spatial = spatial_image_values(maps["original"], rows, args.workers)
        query_spatial = spatial_image_values(maps["query"], rows, args.workers)
        np.testing.assert_array_equal(query_spatial[1], original_spatial[1])
        np.testing.assert_array_equal(query_spatial[2], original_spatial[2])
        seed_points, seed_draws, archive = [], [], {}
        for branch in METHODS:
            values, cases, foreground = query_spatial if branch in ("query", "aggregation_only") else original_spatial
            pp, dd = [], []
            for metric in range(2):
                point, draw = base.spatial_group_bootstrap(values[:, metric:metric+1], cases, expected["group_ids"], plan)
                pp.append(point)
                dd.append(draw)
            seed_points.append(np.stack(pp, axis=-1))
            seed_draws.append(np.stack(dd, axis=-1))
            archive[branch] = values
        space_points.append(np.stack(seed_points))
        space_draws.append(np.stack(seed_draws, axis=1))
        np.savez_compressed(stats / "spatial_per_image.npz", **archive, cases=cases, foreground_pixels=foreground,
            **{key: expected[key] for key in ("image_paths", "class_names", "group_ids")},
            metric_names=np.asarray(["pixel_auroc", "mask_pg"]))
        audits.append({"support_seed": seed, "native_map_equalities": "exact",
                       "Original_across_seeds": "exact", "S_score_role": "support-only fitted final-score residual",
                       "S_spatial_role": "retains Original, does not explain S residual"})
    cp, cd = np.stack(class_points), np.stack(class_draws)
    sp, sd = np.stack(space_points), np.stack(space_draws)
    classification = base.legacy.summarize_arrays("AlphaRAD", DATASET, cp, cd, catalog["class_names"], UNIT, methods=METHODS)
    spatial = base.legacy.summarize_arrays("AlphaRAD", DATASET, sp, sd, catalog["class_names"], UNIT,
        methods=METHODS, metric_names=("pixel_auroc", "mask_pg"))
    filenames = {"per_seed": "PER_SEED_RESULTS.csv", "means": "MEAN_SD_COMPARISON.csv", "per_class": "PER_CLASS_MEAN_SD.csv",
                 "comparisons": "PAIRED_COMPARISONS.csv", "seed_comparisons": "PER_SEED_PAIRED_COMPARISONS.csv",
                 "class_comparisons": "PER_CLASS_PAIRED_COMPARISONS.csv"}
    for key, name in filenames.items():
        base.write_csv(out / name, classification[key])
        base.write_csv(out / ("SPATIAL_" + name), spatial[key])
    base.write_csv(out / "FACTORIAL_INTERACTION.csv", [base.interaction_summary(DATASET, cp, cd, base.legacy.METRIC_NAMES, UNIT)])
    base.write_csv(out / "SPATIAL_FACTORIAL_INTERACTION.csv", [base.interaction_summary(DATASET, sp, sd, ("pixel_auroc", "mask_pg"), UNIT)])
    np.savez_compressed(out / "spatial_bootstrap_metrics.npz", point_metrics=sp, replicate_metrics=sd,
        method_names=np.asarray(METHODS), metric_names=np.asarray(["pixel_auroc", "mask_pg"]), draw_plan_sha256=np.asarray(plan.sha256))
    summary = {"status": "complete", "created_at": datetime.now().isoformat(), "dataset": DATASET,
        "test_images": len(rows), "test_groups": len(plan.patient_ids), "test_positive_images": int(expected["labels"].sum()),
        "grouping": UNIT, "patient_identity_verified": False, "external_hospital_validation": False,
        "support_seeds": list(SEEDS), "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "uncertainty_note": base.legacy.UNCERTAINTY_NOTE,
        "scope": "new same-source holdout under predeclared exclusions; not new-hospital or verified patient-independent validation",
        "spatial_definition": "positive-image pixel AUROC (image-equal mean) and peak-in-union-mask PG; raw native map without CLS bilinearly resized 37x37 to1024x1024, align_corners=False; relative-gap column-major RLE union; no threshold fitting",
        "map_limitation": "Q=Aggregation and Original=Readout maps exactly by construction; S retains Original map and has no adapted native map",
        "source_sha256": sources, "classification": classification["detail"], "spatial": spatial["detail"],
        "audits": audits, "coverage": {"classification_seed_rows": len(classification["per_seed"]),
            "classification_means": len(classification["means"]), "spatial_seed_rows": len(spatial["per_seed"]), "spatial_means": len(spatial["means"])}}
    spatial_means = {r["method"]: r for r in spatial["means"]}
    spatial_pairs = {(r["target"], r["reference"]): r for r in spatial["comparisons"]}
    lines = ["AlphaRAD / new same-source SIIM holdout: five support seeds, native-query path intervention", "",
        "This cohort is not an independent hospital or verified patient-isolated cohort. Bootstrap unit is DICOM identifier group.",
        "Mean±sample SD across seeds 42–46; paired conditional test-group bootstrap, all five supports fixed; no threshold selection.",
        "Native maps satisfy Q=Aggregation and Original=Readout exactly. S retains Original map and does not gain an adapted explanation.", "",
        "| Method / contrast | AUROC | AUPRC | Positive-image pixel AUROC | Mask PG |", "|---|---:|---:|---:|---:|"]
    for row in classification["means"]:
        sr = spatial_means[row["method"]]
        values = [f"{row[m+'_mean_percent']:.2f} ± {row[m+'_sd_pp']:.2f}" for m in ("auroc", "auprc")]
        values += [f"{sr[m+'_mean_percent']:.2f} ± {sr[m+'_sd_pp']:.2f}" for m in ("pixel_auroc", "mask_pg")]
        lines.append("| " + " | ".join([base.DISPLAY[row["method"]], *values]) + " |")
    for row in classification["comparisons"]:
        if row["target"] != "query":
            continue
        sr = spatial_pairs[(row["target"], row["reference"])]
        values = [base.legacy.ci_text(row, m) for m in ("auroc", "auprc")]
        values += [base.legacy.ci_text(sr, m) for m in ("pixel_auroc", "mask_pg")]
        lines.append("| " + " | ".join(["Q − " + base.DISPLAY[row["reference"]], *values]) + " |")
    lines += ["", "Absolute values are percent; contrast confidence intervals are percentage points.",
        "Pixel AUROC is computed separately in every positive image, then averaged; it is not a pooled-pixel AUROC. Mask PG uses one row-major argmax on the same raw interpolated map.",
        "Spatial evaluation only covers positive images. It does not measure normal-image false-positive segmentation, calibrate spatial probabilities, or prove faithful classification attribution.",
        "S comparison uses saved pre-sigmoid margins to avoid compression ties; Q and other native branches use their native sigmoid outputs."]
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    (out / "PAPER_TABLE.md").write_text("\n".join(lines) + "\n")
    for filename, digest in sources.items():
        if base.sha(filename) != digest:
            raise ValueError(f"Evaluation input/source changed during execution: {filename}")
    base.write_json(out / "RESULTS.json", summary)
    base.write_json(out / "AUDIT.json", {"status": "passed", "source_sha256": sources,
        "artifacts": {str(path): base.sha(path) for path in out.iterdir() if path.is_file() and path.name != "AUDIT.json"}})
    print(json.dumps({"status": "complete", "output": str(out), "coverage": summary["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
