#!/usr/bin/env python3
"""Run a portable frozen RadZero query adapter on fixed tests or real images.

This general inference entry has no overnight deadline. It performs no fitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import numpy as np
import torch
from torch.utils.data import DataLoader

from batch_metrics import compute_metrics
from i2cl_data import load_spec
from infer_batch import MODEL_DIR, TestImages, checkpoint_identity, load_model, now
from overnight_query_refinement import eta_name
from query_adapter import DATASETS, DEFAULT_ROOT, QueryAdapter, sha256, write_json


def verify_base(adapter):
    if checkpoint_identity() != adapter.metadata["checkpoint"]:
        raise ValueError("Adapter requires a different frozen RadZero checkpoint")
    for filename, key in (("config.json", "model_config_sha256"),
                          ("preprocessor_config.json", "preprocessor_sha256")):
        if sha256(MODEL_DIR / filename) != adapter.metadata[key]:
            raise ValueError(f"Frozen model/preprocessor differs: {filename}")
    for recorded_path, expected in adapter.metadata["base_source_sha256"].items():
        source = Path(recorded_path)
        if source.parent.name == "model":
            target = MODEL_DIR / source.name
        else:
            target = Path(__file__).with_name(source.name)
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Frozen query/scoring source differs: {target}")


def compare_reference(adapter, original, refined, count):
    """Strict optional comparison to the already-completed run, not training."""
    source = Path(adapter.metadata["source_query_run"])
    if not source.is_dir():
        return {"available": False, "reason": "Original experiment directory not available on this machine"}
    manifest_path = source / "manifest.json"
    if sha256(manifest_path) != adapter.metadata["source_query_manifest_sha256"]:
        raise ValueError("Original query experiment manifest changed")
    manifest = json.loads(manifest_path.read_text())
    errors = {}
    for eta, actual in ((0., original), (adapter.eta, refined)):
        record = manifest["predictions"][eta_name(eta)]
        path = source / record["directory"] / "logits.npy"
        if sha256(path) != record["logits_sha256"]:
            raise ValueError("Original experiment logits changed")
        expected = np.load(path, mmap_mode="r", allow_pickle=False)[:count]
        if expected.shape != actual.shape:
            raise ValueError("Reference prediction count/classes differ")
        error = float(np.abs(actual.astype(float) - expected).max(initial=0))
        if error > 1e-6:
            raise ValueError(f"Real-image adapter mismatch for eta={eta}: {error} > 1e-6")
        errors[str(eta)] = {"max_abs_error": error, "bitwise_equal": bool(np.array_equal(actual, expected))}
    return {"available": True, "passed": True, "atol": 1e-6, "rtol": 0., "eta_checks": errors}


def predict_images(adapter, paths, root, dicom, model, processor, args, output):
    matcher = adapter.make_matcher(model)
    images = TestImages(paths, root, processor, dicom)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                   drop_last=False, pin_memory=args.device.startswith("cuda"))
    if args.workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    original = np.empty((len(paths), len(adapter.class_names)), dtype=np.float32)
    refined = np.empty_like(original)
    maps_directory = output / "maps"
    if args.save_maps:
        maps_directory.mkdir()
    offset, exported_maps = 0, 0
    with torch.inference_mode():
        for pixels, sizes, indices in DataLoader(images, **options):
            stop = offset + len(pixels)
            if not np.array_equal(indices.numpy(), np.arange(offset, stop)):
                raise ValueError("Image order changed")
            pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
            need_maps = args.save_maps and exported_maps < args.map_limit
            # Both query branches share exactly one frozen image-encoder pass.
            values = matcher(pixels, return_maps=need_maps)
            original[offset:stop] = values[0.]["logits"].cpu().numpy()
            refined[offset:stop] = values[adapter.eta]["logits"].cpu().numpy()
            if need_maps:
                for index in range(min(len(pixels), args.map_limit - exported_maps)):
                    np.savez_compressed(
                        maps_directory / f"{offset + index:06d}.npz",
                        original_similarity=values[0.]["similarity_scores"][index].cpu().numpy(),
                        refined_similarity=values[adapter.eta]["similarity_scores"][index].cpu().numpy(),
                        original_image_height_width=sizes[index].numpy(),
                        image_index=np.asarray(offset + index),
                    )
                    exported_maps += 1
            offset = stop
            write_json(output / "progress.json", {
                "status": "inference", "images_completed": offset, "images_total": len(paths),
                "maps_exported": exported_maps, "updated_at": now()})
            if offset == len(pixels) or offset % 512 == 0 or offset == len(paths):
                print(f"{adapter.metadata['dataset']}: {offset}/{len(paths)}", flush=True)
    if offset != len(paths) or not np.isfinite(original).all() or not np.isfinite(refined).all():
        raise ValueError("Incomplete or nonfinite inference")
    if adapter.eta == 0:
        np.testing.assert_array_equal(original, refined)
    np.save(output / "baseline_logits.npy", original, allow_pickle=False)
    np.save(output / "predictions.npy", refined, allow_pickle=False)
    return original, refined, exported_maps


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; expose a free GPU or use --device cpu")
    custom = bool(args.images)
    names = ["custom_images"] if custom else args.datasets
    results = {}
    model = processor = None
    try:
        for name in names:
            adapter_path = args.adapter or args.adapter_root / name / "query_adapter.npz"
            adapter = QueryAdapter.load(adapter_path)
            verify_base(adapter)
            spec = None
            if custom:
                paths = [str(path.expanduser().resolve()) for path in args.images]
                if any(not Path(path).is_file() for path in paths):
                    raise FileNotFoundError("One or more requested images do not exist")
                root, dicom = Path("/"), args.dicom
            else:
                if adapter.metadata["dataset"] != name:
                    raise ValueError("Adapter belongs to another dataset")
                split_dir = args.split_dir or Path(adapter.metadata["split_dir"])
                spec = load_spec(split_dir, name, "test")
                expected = [value for path, value in adapter.metadata["split_sha256"].items()
                            if Path(path).name == "test.csv"]
                if len(expected) != 1 or sha256(split_dir / name / "test.csv") != expected[0]:
                    raise ValueError("Test split differs from adapter provenance")
                if tuple(spec["class_names"]) != adapter.class_names or tuple(spec["prompts"]) != adapter.prompts:
                    raise ValueError("Test class/prompt order differs from exported queries")
                paths = spec["image_paths"][:args.limit_test] if args.limit_test else spec["image_paths"]
                root, dicom = Path(spec["data_root"]), name in ("SIIM", "RSNA")
            if not paths:
                raise ValueError("No images to evaluate")
            directory = args.output / name
            directory.mkdir()
            write_json(directory / "run_config.json", {
                "dataset": name, "adapter": str(adapter.path), "adapter_sha256": sha256(adapter.path),
                "eta": adapter.eta, "adapter_provenance": adapter.metadata,
                "image_paths": paths, "class_names": list(adapter.class_names), "prompts": list(adapter.prompts),
                "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "batch_size": args.batch_size, "workers": args.workers, "image_count": len(paths),
                "is_smoke": bool(args.limit_test and not custom), "fitting_performed": False,
                "precision": "float32; TF32 disabled", "model_frozen": True,
                "source_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                                  str(Path(__file__).with_name("query_adapter.py")):
                                  sha256(Path(__file__).with_name("query_adapter.py"))},
            })
            if model is None:
                model, processor, _ = load_model(torch.device(args.device))
                model = model.float().requires_grad_(False).eval()
            original, refined, maps = predict_images(
                adapter, paths, root, dicom, model, processor, args, directory)
            with (directory / "predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "image_path",
                                 *[f"base_logit:{c}" for c in adapter.class_names],
                                 *[f"adapted_logit:{c}" for c in adapter.class_names]])
                writer.writerows([index, path, *base.tolist(), *adapted.tolist()]
                                 for index, (path, base, adapted) in enumerate(zip(paths, original, refined)))
            result = {
                "status": "complete", "image_count": len(paths), "eta": adapter.eta,
                "is_smoke": bool(args.limit_test and not custom), "maps_exported": maps,
                "map_semantics": "Raw query-conditioned similarity on the original patch grid; not a thresholded segmentation mask.",
                "baseline_logits_sha256": sha256(directory / "baseline_logits.npy"),
                "predictions_sha256": sha256(directory / "predictions.npy"),
            }
            if spec is not None:
                result["reference_check"] = compare_reference(adapter, original, refined, len(paths))
                score_spec = {**spec, "image_paths": paths, "labels": spec["labels"][:len(paths)]}
                result["baseline_metrics"] = compute_metrics(score_spec, original)
                result["metrics"] = compute_metrics(score_spec, refined)
                result["delta_pp"] = {
                    key: None if result["metrics"][key] is None or result["baseline_metrics"][key] is None
                    else 100 * (result["metrics"][key] - result["baseline_metrics"][key])
                    for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")}
            write_json(directory / "result.json", result)
            write_json(directory / "progress.json", {"status": "complete", "updated_at": now()})
            results[name] = result
            write_json(args.output / "summary.json", {"status": "running", "datasets": results})
        write_json(args.output / "summary.json", {"status": "complete", "datasets": results})
    except BaseException as error:
        write_json(args.output / "failure.json", {
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--adapter", type=Path, help="Explicit query_adapter.npz or its directory")
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--images", type=Path, nargs="+", help="Unlabelled PNG/JPG/DICOM paths; requires --adapter")
    parser.add_argument("--dicom", action="store_true", help="Decode all custom --images as DICOM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=0, help="First N verified test images, for smoke checks only")
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--map-limit", type=int, default=8, help="Save raw original/refined maps for first N images")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0 or args.limit_test < 0 or args.map_limit < 1:
        parser.error("Invalid batch/workers/image/map limit")
    if args.images and not args.adapter:
        parser.error("--images requires --adapter")
    if args.adapter and not args.images and len(args.datasets) != 1:
        parser.error("An explicit --adapter requires exactly one --datasets entry")
    args.output = args.output.expanduser().resolve()
    if args.split_dir:
        args.split_dir = args.split_dir.expanduser().resolve()
    return args


if __name__ == "__main__":
    run(parse_args())
