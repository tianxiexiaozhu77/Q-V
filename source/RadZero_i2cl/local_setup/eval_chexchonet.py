#!/usr/bin/env python3
"""Evaluate frozen RadZero and the fixed phase6 query/center80 method.

This is a new, independently frozen CheXchoNet protocol. Its catalog supplies
the task definitions, reconstructed prompts, support CSVs, and patient-disjoint
paired test. It does not change any historical RadZero experiment or use test
labels to construct queries. No optimizer/backpropagation is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
GEOMETRY_PATH = (WORKSPACE / "experiments/query_search_20261021_152010"
                 / "code_phase6/view_geometry.py")
TASKS = ("slvh", "dlv", "composite_slvh_dlv")


def center_crop_view(pixels, ratio):
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratio must lie in (0, 1]")
    h, w = pixels.shape[-2:]
    ch, cw = int(h * ratio), int(w * ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    return torch.nn.functional.interpolate(pixels[..., top:top + ch, left:left + cw],
        size=(h, w), mode="bilinear", align_corners=False, antialias=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def read_rows(path, image_root):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        image = Path(row.get("image_path", row.get("image", "")))
        if not str(image) or str(image) == ".":
            raise ValueError(f"Missing image path in {path}")
        row["image_path"] = str((image_root / image).resolve())
        if not row.get("patient_id"):
            raise ValueError(f"Missing patient_id in {path}")
        if "composite" not in row and "composite_slvh_dlv" in row:
            row["composite"] = row["composite_slvh_dlv"]
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate images in {path}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full-original", "paired-four-arm"), required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--gpu", type=int, help="Physical GPU index; sets CUDA_VISIBLE_DEVICES")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eta", type=float, default=.3)
    parser.add_argument("--method", choices=("closed_form", "uncertainty"), default="closed_form")
    parser.add_argument("--covariance-shrinkage", type=float, default=.2)
    parser.add_argument("--covariance-ridge", type=float, default=1e-3)
    parser.add_argument("--reliability-power", type=float, default=1.0)
    parser.add_argument("--crop-ratio", type=float, default=.8)
    parser.add_argument("--view-weight", type=float, default=.5)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only: first N full-test images")
    parser.add_argument("--validate-only", action="store_true", help="Check manifests without loading model/CUDA")
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.workers < 0 or args.limit < 0 or args.eta < 0
            or not 0 < args.crop_ratio <= 1 or not 0 <= args.view_weight <= 1
            or not 0 <= args.covariance_shrinkage <= 1 or args.covariance_ridge < 0
            or args.reliability_power < 0):
        parser.error("Invalid batch size, workers, limit or eta")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Duplicate tasks")
    args.catalog = args.catalog.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    return args


def load_catalog(args):
    catalog = json.loads(args.catalog.read_text())
    base = args.catalog.parent
    expected_hashes = {}
    if "datasets" in catalog:
        prompts_path = base / "prompts.json"
        prompts = json.loads(prompts_path.read_text())
        converted = {"tasks": {}, "data_root": "/", "prompt_provenance": prompts}
        for task in TASKS:
            entry = catalog["datasets"]["chexchonet_" + task]
            converted["tasks"][task] = {
                **prompts["tasks"][task], "support_csv": entry["support_csv_path"],
                "label_column": entry["source_label_column"], "catalog_entry": entry}
            for path_key, hash_key in (("support_csv_path", "support_csv_sha256"),
                                       ("csv_path", "csv_sha256"),
                                       ("original_csv_path", "original_csv_sha256")):
                expected_hashes[entry[path_key]] = entry[hash_key]
        converted["full_test_csv"] = str(base / "full_evaluation.csv")
        converted["paired_test_csv"] = str(base / "evaluation.csv")
        catalog = converted
    image_root = Path(catalog.get("data_root", catalog.get("image_root", "/")))
    if not image_root.is_absolute():
        image_root = (base / image_root).resolve()

    def resolve(value):
        return (base / str(value)).resolve()

    full_csv = resolve(catalog["full_test_csv"])
    paired_csv = resolve(catalog["paired_test_csv"])
    full = read_rows(full_csv, image_root)
    paired = read_rows(paired_csv, image_root)
    full_lookup = {row["image_path"]: row for row in full}
    if any(row["image_path"] not in full_lookup for row in paired):
        raise ValueError("Paired test is not a subset of full test")
    paired_patients = {row["patient_id"] for row in paired}
    supports, task_config = {}, {}
    hashes = {str(args.catalog): sha256(args.catalog),
              str(full_csv): sha256(full_csv), str(paired_csv): sha256(paired_csv)}
    if (base / "prompts.json").is_file():
        hashes[str(base / "prompts.json")] = sha256(base / "prompts.json")
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Catalog checksum differs: {path}")
        hashes[path] = actual
    # Validate ALL support tasks, including other tasks not selected for this shard.
    for task, config in catalog["tasks"].items():
        if task not in TASKS:
            continue
        csv_path = resolve(config["support_csv"])
        rows = read_rows(csv_path, image_root)
        if paired_patients & {row["patient_id"] for row in rows}:
            raise ValueError(f"Support/test patient overlap in {task}")
        label_column = config.get("label_column", task)
        labels = [int(row[label_column]) for row in rows]
        if set(labels) != {0, 1}:
            raise ValueError(f"Support must contain both binary labels: {task}")
        if len(set(row["patient_id"] for row in rows)) != len(rows):
            raise ValueError(f"Support must use distinct patients: {task}")
        if labels.count(0) != 5 or labels.count(1) != 5:
            raise ValueError(f"Frozen protocol requires five positives and five negatives: {task}")
        prompt = config.get("radzero_prompt", config.get("prompt"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing explicit RadZero prompt: {task}")
        for row in rows + paired:
            if int(row[label_column]) not in (0, 1):
                raise ValueError(f"Non-binary labels: {task}")
        supports[task] = rows
        task_config[task] = {**config, "label_column": label_column, "prompt": prompt}
        hashes[str(csv_path)] = sha256(csv_path)
    for task in args.tasks:
        if task not in task_config:
            raise ValueError(f"Missing task in catalog: {task}")
        label = task_config[task]["label_column"]
        for row in paired:
            source = full_lookup[row["image_path"]]
            if row["patient_id"] != source["patient_id"] or row[label] != source[label]:
                raise ValueError(f"Full/paired identity or label mismatch: {task}")
    paths = {row["image_path"] for row in full}
    for rows in supports.values():
        paths.update(row["image_path"] for row in rows)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return catalog, full, paired, supports, task_config, hashes


def run(args):
    # Set physical GPU visibility before importing Torch or any model module.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "4")
    catalog, full, paired, supports, task_config, hashes = load_catalog(args)
    if args.validate_only:
        print(json.dumps({"status": "validated", "full_images": len(full),
                          "paired_images": len(paired), "tasks": args.tasks,
                          "support_images": {task: len(rows) for task, rows in supports.items()},
                          "gpu_used": False}, indent=2))
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from batch_metrics import compute_metrics
    from extract_frozen_features import FrozenFeatureExtractor
    from infer_batch import TestImages, checkpoint_identity, load_model, now, MODEL_DIR
    from overnight_query_refinement import (fit_query_directions,
        fit_uncertainty_query_directions, QueryRefinementMatcher,
        UNCERTAINTY_VERSION, VERSION)

    geometry_spec = importlib.util.spec_from_file_location("chexcho_phase6_geometry", GEOMETRY_PATH)
    geometry = importlib.util.module_from_spec(geometry_spec)
    geometry_spec.loader.exec_module(geometry)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        started = time.monotonic()
        tasks = args.tasks
        classes = len(tasks)
        prompts = [task_config[task]["prompt"] for task in tasks]
        full = full if args.mode == "full-original" else paired
        full = full[:args.limit] if args.limit else full
        full_positions = {row["image_path"]: i for i, row in enumerate(full)}
        paired = [row for row in paired if row["image_path"] in full_positions]
        if not paired:
            raise ValueError("No paired-test rows remain after smoke limit")
        support_rows = {}
        for task in tasks:
            for row in supports[task]:
                support_rows.setdefault(row["image_path"], row)
        support_rows = list(support_rows.values())
        support_positions = {row["image_path"]: i for i, row in enumerate(support_rows)}
        support_paths = [row["image_path"] for row in support_rows]
        source_paths = [Path(__file__).resolve(), GEOMETRY_PATH,
                        Path(__file__).with_name("extract_frozen_features.py"),
                        Path(__file__).with_name("overnight_query_refinement.py"),
                        Path(__file__).with_name("infer_batch.py"),
                        Path(__file__).with_name("batch_metrics.py")]
        metadata = {"model": "RadZero", "dataset": "CheXchoNet custom 7494",
                    "tasks": tasks, "task_definitions": {task: task_config[task] for task in tasks},
                    "catalog": str(args.catalog), "catalog_sha256": sha256(args.catalog),
                    "mode": args.mode, "prompt_provenance": catalog.get("prompt_provenance"),
                    "manifest_sha256": hashes, "source_sha256": {str(p): sha256(p) for p in source_paths},
                    "checkpoint": checkpoint_identity(), "model_directory": str(MODEL_DIR),
                    "eta": args.eta, "crop_ratio": args.crop_ratio, "view_weight": args.view_weight,
                    "preprocessing": "Released RadZero PNG/JPG minmax normalization + local Blip processor 518",
                    "fusion": "weighted native-logit average before sigmoid",
                    "query_fit": ("Per task: support covariance-shrinkage Mahalanobis direction with reliability weighting"
                                  if args.method == "uncertainty" else
                                  "Per task: normalized positive-minus-negative mean of normalized native class-local support contexts; q'=normalize(normalize(q)+eta*d)"),
                    "method": args.method,
                    "version": UNCERTAINTY_VERSION if args.method == "uncertainty" else VERSION,
                    "covariance_shrinkage": args.covariance_shrinkage,
                    "covariance_ridge": args.covariance_ridge,
                    "reliability_power": args.reliability_power,
                    "support_counts": {task: len(supports[task]) for task in tasks},
                    "full_count": len(full), "paired_count": len(paired),
                    "gpu": args.gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "batch_size": args.batch_size, "workers": args.workers,
                    "precision": "float32; TF32 disabled", "model_frozen": True,
                    "gradient_optimization": False, "test_labels_used_for_query_fit": False,
                    "is_smoke": bool(args.limit), "created_at": now()}
        write_json(args.output / "run_config.json", metadata)
        model, processor, tokenizer = load_model(torch.device(args.device))
        model = model.float().requires_grad_(False).eval()
        extractor = FrozenFeatureExtractor(model, tokenizer, prompts)

        def loader(rows):
            paths = [row["image_path"] for row in rows]
            dataset = TestImages(paths, Path("/"), processor, False)
            options = dict(batch_size=args.batch_size, num_workers=args.workers,
                           shuffle=False, drop_last=False,
                           pin_memory=args.device.startswith("cuda"))
            if args.workers:
                options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
            return DataLoader(dataset, **options)

        support_features = np.empty((len(support_rows), classes, extractor.text_features.shape[-1]), dtype=np.float32)
        official_check = None
        with torch.inference_mode():
            for pixels, _, indices in loader(support_rows if args.mode == "paired-four-arm" else full[:min(2, len(full))]):
                pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
                if official_check is None:
                    official_check = extractor.verify_official_api(pixels)
                if args.mode == "paired-four-arm":
                    support_features[indices.numpy()] = extractor(pixels)["class_local"].cpu().numpy()
        original_queries = extractor.text_features.detach().cpu().numpy().copy()
        refined_queries = np.empty_like(original_queries)
        fits = {}
        refined_queries[:] = original_queries
        for j, task in enumerate(tasks if args.mode == "paired-four-arm" else []):
            rows = supports[task]
            positions = [support_positions[row["image_path"]] for row in rows]
            labels = np.asarray([[int(row[task_config[task]["label_column"]])] for row in rows], dtype=np.int8)
            fit_etas = (0.,) if args.eta == 0.0 else (0., args.eta)
            if args.method == "uncertainty":
                fit = fit_uncertainty_query_directions(
                    support_features[positions, j:j+1], labels,
                    original_queries[j:j+1], fit_etas,
                    covariance_shrinkage=args.covariance_shrinkage,
                    ridge=args.covariance_ridge,
                    reliability_power=args.reliability_power)
            else:
                fit = fit_query_directions(support_features[positions, j:j+1], labels,
                                           original_queries[j:j+1], fit_etas)
            refined_queries[j] = fit["queries"][-1, 0]
            fits[task] = fit
            np.savez_compressed(args.output / f"support_fit_{task}.npz", **fit,
                                support_image_paths=np.asarray([row["image_path"] for row in rows]))
        np.savez_compressed(args.output / "query_adapter.npz",
                            original_queries=original_queries, refined_queries=refined_queries,
                            tasks=np.asarray(tasks), prompts=np.asarray(prompts))
        write_json(args.output / "reference_equivalence.json", official_check)
        query_values = (np.stack((original_queries, refined_queries))
                        if args.mode == "paired-four-arm" and args.eta != 0.0
                        else original_queries[None])
        etas = fit_etas if args.mode == "paired-four-arm" else (0.,)
        matcher = QueryRefinementMatcher(model, query_values, etas)
        baseline = np.full((len(full), classes), np.nan, dtype=np.float32)
        query_only = np.full((len(full), classes), np.nan, dtype=np.float32)
        center_only = np.full((len(full), classes), np.nan, dtype=np.float32)
        original_center = np.full((len(full), classes), np.nan, dtype=np.float32)
        paired_set = {row["image_path"] for row in paired}
        paired_indices = np.asarray([full_positions[row["image_path"]] for row in paired], dtype=np.int64)
        chunks = args.output / "batches"
        chunks.mkdir()
        offset = 0
        inference_started = time.monotonic()
        with torch.inference_mode():
            for pixels, _, indices in loader(full):
                idx = indices.numpy()
                if not np.array_equal(idx, np.arange(offset, offset + len(idx))):
                    raise ValueError("Evaluation order changed")
                pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
                values = matcher(pixels)
                baseline[idx] = values[0.]["logits"].cpu().numpy()
                keep = np.asarray([full[int(i)]["image_path"] in paired_set for i in idx])
                if args.mode == "paired-four-arm" and keep.any():
                    selected = torch.as_tensor(keep, device=args.device)
                    query_only[idx[keep]] = values[args.eta]["logits"][selected].cpu().numpy()
                    cropped = center_crop_view(pixels[selected], args.crop_ratio)
                    vision = matcher.prepare_vision(model.forward_vision_model(cropped)["vision_tokens"])
                    original_center[idx[keep]] = matcher.match_prepared(vision, matcher.queries[0])["logits"].cpu().numpy()
                    refined_query = matcher.queries[1] if args.eta != 0.0 else matcher.queries[0]
                    center_only[idx[keep]] = matcher.match_prepared(vision, refined_query)["logits"].cpu().numpy()
                offset += len(idx)
                np.savez_compressed(chunks / f"batch_{int(idx[0]):06d}_{offset:06d}.npz",
                                    indices=idx, baseline=baseline[idx], query_only=query_only[idx],
                                    center_only=center_only[idx], original_center=original_center[idx], paired_mask=keep)
                elapsed = time.monotonic() - inference_started
                write_json(args.output / "progress.json", {
                    "status": "inference", "completed_images": offset, "total_images": len(full),
                    "images_per_second": offset / max(elapsed, .001), "updated_at": now()})
                if offset == len(idx) or offset % (args.batch_size * 20) == 0 or offset == len(full):
                    print(f"RadZero {','.join(tasks)}: {offset}/{len(full)} ({offset/max(elapsed,.001):.2f} images/s)", flush=True)
        refined = ((1.0 - args.view_weight) * query_only[paired_indices]
                   + args.view_weight * center_only[paired_indices])
        if not np.isfinite(baseline).all() or (args.mode == "paired-four-arm" and not np.isfinite(refined).all()):
            raise ValueError("Incomplete/nonfinite predictions")
        baseline_paired = baseline[paired_indices]
        np.save(args.output / "baseline_full_logits.npy", baseline, allow_pickle=False)
        np.save(args.output / "baseline_paired_logits.npy", baseline_paired, allow_pickle=False)
        if args.mode == "paired-four-arm":
            np.save(args.output / "refined_paired_logits.npy", refined, allow_pickle=False)
            np.save(args.output / "query_only_paired_logits.npy", query_only[paired_indices], allow_pickle=False)
            np.save(args.output / "center_only_paired_logits.npy", center_only[paired_indices], allow_pickle=False)
            original_tta = ((1.0 - args.view_weight) * baseline_paired
                            + args.view_weight * original_center[paired_indices])
            np.save(args.output / "original_tta_paired_logits.npy", original_tta, allow_pickle=False)

        def metrics(rows, logits):
            labels = np.asarray([[int(row[task_config[task]["label_column"]]) for task in tasks] for row in rows])
            return compute_metrics({"name": "CheXchoNet custom 7494", "class_names": tasks,
                                    "image_paths": [row["image_path"] for row in rows], "labels": labels}, logits)

        full_metrics = metrics(full, baseline)
        base_metrics = metrics(paired, baseline_paired)
        ours_metrics = metrics(paired, refined) if args.mode == "paired-four-arm" else None
        results = {"status": "complete", "model": "RadZero", "tasks": tasks,
                   "mode": args.mode,
                   "is_smoke": bool(args.limit), "full_count": len(full), "paired_count": len(paired),
                   "baseline_full": full_metrics if args.mode == "full-original" else None, "baseline_paired": base_metrics,
                   "ours_paired": ours_metrics, "eta": args.eta,
                   "delta_pp": {task: {metric: None if ours_metrics is None or base_metrics["per_class"][task][metric] is None or ours_metrics["per_class"][task][metric] is None else
                                         100 * (ours_metrics["per_class"][task][metric] - base_metrics["per_class"][task][metric])
                                        for metric in ("auroc", "auprc")} for task in tasks},
                   "elapsed_seconds": time.monotonic() - started, "completed_at": now()}
        if args.mode == "paired-four-arm":
            results["arms"] = {
                "original": base_metrics,
                "query_only": metrics(paired, query_only[paired_indices]),
                "tta_center_original": metrics(paired, original_tta),
                "tta_center_refined": ours_metrics,
            }
        write_json(args.output / "result.json", results)
        with (args.output / "predictions_paired.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "patient_id", *[f"label:{t}" for t in tasks],
                             *[f"baseline_logit:{t}" for t in tasks],
                             *([f"refined_logit:{t}" for t in tasks] if args.mode == "paired-four-arm" else [])])
            for i, row in enumerate(paired):
                writer.writerow([row["image_path"], row["patient_id"],
                                 *[row[task_config[t]["label_column"]] for t in tasks],
                                 *baseline_paired[i].tolist(),
                                 *(refined[i].tolist() if args.mode == "paired-four-arm" else [])])
        with (args.output / "predictions_full.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "patient_id", *[f"label:{t}" for t in tasks],
                             *[f"baseline_logit:{t}" for t in tasks]])
            for i, row in enumerate(full):
                writer.writerow([row["image_path"], row["patient_id"],
                                 *[row[task_config[t]["label_column"]] for t in tasks], *baseline[i].tolist()])
        write_json(args.output / "progress.json", {"status": "complete", "updated_at": now()})
        print(json.dumps(results, indent=2), flush=True)
    except BaseException as exc:
        write_json(args.output / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    run(parse_args())
