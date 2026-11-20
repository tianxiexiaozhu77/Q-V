#!/usr/bin/env python3
"""Eta/crop sensitivity for RadZero on the frozen CheXpert holdout.

This fills the one missing combination in the older campaign: RadZero had a
verified CheXchoNet path but no CheXpert query-transfer worker.  The script
uses the same CheXpert support/test manifests as the AlphaRAD evaluation,
keeps the released RadZero scorer unchanged, and computes every declared eta
and center-crop ratio in one model load.  Test labels are read only after all
predictions have been written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
RAD = ROOT / "RadZero_i2cl"
sys.path.insert(0, str(RAD / "local_setup"))
from batch_metrics import compute_metrics  # noqa: E402
from extract_frozen_features import FrozenFeatureExtractor  # noqa: E402
from infer_batch import TestImages, checkpoint_identity, load_model  # noqa: E402
from overnight_query_refinement import (  # noqa: E402
    QueryRefinementMatcher, fit_query_directions, fit_uncertainty_query_directions,
    validate_etas, UNCERTAINTY_VERSION, VERSION,
)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n")
    tmp.replace(path)


def crop_view(pixels, ratio):
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratios must lie in (0, 1]")
    h, w = pixels.shape[-2:]
    ch, cw = int(h * ratio), int(w * ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    return F.interpolate(pixels[..., top:top + ch, left:left + cw], size=(h, w),
                         mode="bilinear", align_corners=False, antialias=False)


def loader(paths, processor, batch_size, workers, device):
    dataset = TestImages(paths, Path("/"), processor, False)
    options = dict(batch_size=batch_size, shuffle=False, num_workers=workers,
                   drop_last=False, pin_memory=device.startswith("cuda"))
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True,
                       prefetch_factor=2)
    return DataLoader(dataset, **options)


def main(args):
    args.etas = validate_etas(args.etas)
    args.crop_ratios = tuple(float(x) for x in args.crop_ratios)
    if any(not math.isfinite(x) or not 0.0 < x <= 1.0 for x in args.crop_ratios):
        raise ValueError("crop ratios must be finite values in (0, 1]")
    if not 0.0 <= args.view_weight <= 1.0:
        raise ValueError("view-weight must lie in [0, 1]")
    catalog = json.loads(args.catalog.read_text())
    entry = catalog["datasets"]["chexpert"]
    support_manifest = json.loads(Path(entry["support_manifest"]).read_text())
    test_manifest = json.loads(Path(entry["manifest"]).read_text())
    # query_transfer's test manifest is the matched phase6 JSON with rows;
    # the support manifest stores the exact same ordered test-manifest path.
    if not test_manifest.get("rows"):
        test_manifest = json.loads(Path(support_manifest["test_manifest"]).read_text())
    classes = list(entry["class_names"])
    supports = support_manifest["support_rows"]
    rows = test_manifest["rows"]
    if args.limit:
        rows = rows[:args.limit]
    if len(classes) != 12 or len(supports) != 60:
        raise ValueError("Unexpected frozen CheXpert support/class count")
    support_paths = [r["image"] for r in supports]
    support_labels = np.asarray([r["labels"] for r in supports], dtype=np.uint8)
    test_paths = [r["image"] for r in rows]
    test_labels = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
    if set(support_paths) & set(test_paths):
        raise ValueError("Support/test image overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", {
        "model": "RadZero", "dataset": "CheXpert", "etas": list(args.etas),
        "crop_ratios": list(args.crop_ratios), "view_weight": args.view_weight,
        "method": args.method,
        "covariance_shrinkage": args.covariance_shrinkage,
        "covariance_ridge": args.covariance_ridge,
        "reliability_power": args.reliability_power,
        "support_count": len(supports), "test_count": len(rows),
        "test_labels_used_for_fit": False, "network_weights_frozen": True,
    })
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, processor, tokenizer = load_model(torch.device(args.device))
    model = model.float().requires_grad_(False).eval()
    prompts = [f"There is {name}." for name in classes]
    extractor = FrozenFeatureExtractor(model, tokenizer, prompts)
    with torch.inference_mode():
        support_features = np.empty((len(supports), len(classes), extractor.text_features.shape[-1]), np.float32)
        seen = 0
        for pixels, _, indices in loader(support_paths, processor, args.batch_size, args.workers, args.device):
            values = extractor(pixels.to(args.device, dtype=torch.float32, non_blocking=True))
            support_features[indices.numpy()] = values["class_local"].cpu().numpy()
            seen += len(indices)
        if seen != len(supports):
            raise ValueError("Support loader did not cover all images")
    original_queries = extractor.text_features.detach().cpu().numpy()
    if args.method == "uncertainty":
        fit = fit_uncertainty_query_directions(
            support_features, support_labels, original_queries, args.etas,
            covariance_shrinkage=args.covariance_shrinkage,
            ridge=args.covariance_ridge,
            reliability_power=args.reliability_power)
    else:
        fit = fit_query_directions(support_features, support_labels,
                                   original_queries, args.etas)
    matcher = QueryRefinementMatcher(model, fit["queries"], args.etas)
    arrays = {(ratio, eta): np.empty((len(rows), len(classes)), np.float32)
              for ratio in args.crop_ratios for eta in args.etas}
    base = {eta: np.empty((len(rows), len(classes)), np.float32) for eta in args.etas}
    offset = 0
    started = time.monotonic()
    with torch.inference_mode():
        for pixels, _, indices in loader(test_paths, processor, args.batch_size, args.workers, args.device):
            idx = indices.numpy()
            if not np.array_equal(idx, np.arange(offset, offset + len(idx))):
                raise ValueError("Test order changed")
            pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
            identity = matcher(pixels)
            for eta in args.etas:
                base[eta][idx] = identity[eta]["logits"].cpu().numpy()
            for ratio in args.crop_ratios:
                center = matcher(crop_view(pixels, ratio))
                for eta in args.etas:
                    values = ((1.0 - args.view_weight) * identity[eta]["logits"]
                              + args.view_weight * center[eta]["logits"])
                    arrays[(ratio, eta)][idx] = values.cpu().numpy()
            offset += len(idx)
            write_json(args.output / "progress.json", {
                "status": "running", "completed_images": offset,
                "total_images": len(rows), "updated_at": time.time()})
    if offset != len(rows):
        raise ValueError("Incomplete test inference")
    spec = {"name": "CheXpert", "class_names": classes,
            "image_paths": test_paths, "labels": test_labels}
    records = []
    for ratio in args.crop_ratios:
        for eta in args.etas:
            metrics = compute_metrics(spec, arrays[(ratio, eta)])
            baseline = compute_metrics(spec, base[eta])
            delta = {key: 100.0 * (metrics[key] - baseline[key])
                     if metrics.get(key) is not None and baseline.get(key) is not None else None
                     for key in ("macro_auroc", "macro_auprc")}
            np.save(args.output / f"logits_r{ratio:g}_e{eta:g}.npy", arrays[(ratio, eta)], allow_pickle=False)
            records.append({"crop_ratio": ratio, "eta": eta, "metrics": metrics,
                            "identity_eta_metrics": baseline, "delta_vs_identity_eta_pp": delta})
    result = {"status": "complete", "model": "RadZero", "dataset": "CheXpert",
              "support_count": len(supports), "test_count": len(rows),
              "etas": list(args.etas), "crop_ratios": list(args.crop_ratios),
              "view_weight": args.view_weight, "prompts": prompts,
              "method": args.method,
              "version": UNCERTAINTY_VERSION if args.method == "uncertainty" else VERSION,
              "covariance_shrinkage": args.covariance_shrinkage,
              "covariance_ridge": args.covariance_ridge,
              "reliability_power": args.reliability_power,
              "records": records, "elapsed_seconds": time.monotonic() - started,
              "test_labels_used_for_query_fit": False, "no_gradient_training": True,
              "checkpoint": checkpoint_identity()}
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {"status": "complete", "completed_images": len(rows)})
    print(json.dumps({"status": "complete", "records": len(records)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--etas", type=float, nargs="+", default=[0., .1, .2, .3, .5, .7])
    parser.add_argument("--crop-ratios", type=float, nargs="+", default=[.6, .7, .8, .9])
    parser.add_argument("--view-weight", type=float, default=.5)
    parser.add_argument("--method", choices=("closed_form", "uncertainty"), default="closed_form")
    parser.add_argument("--covariance-shrinkage", type=float, default=.2)
    parser.add_argument("--covariance-ridge", type=float, default=1e-3)
    parser.add_argument("--reliability-power", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
