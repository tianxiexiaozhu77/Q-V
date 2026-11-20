#!/usr/bin/env python3
"""Frozen RadZero + exported ridge adapter, directly from images at 518x518."""

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
from extract_frozen_features import FrozenFeatureExtractor
from i2cl_data import load_spec
from infer_batch import MODEL_DIR, TestImages, checkpoint_identity, load_model, now
from ridge_adapter import DATASETS, DEFAULT_ADAPTER_ROOT, RidgeAdapter, sha256, write_json


def verify_base(adapter):
    if checkpoint_identity() != adapter.metadata["checkpoint"]:
        raise ValueError("Adapter requires a different frozen checkpoint")
    for filename, key in (("config.json", "model_config_sha256"),
                          ("preprocessor_config.json", "preprocessor_sha256")):
        if sha256(MODEL_DIR / filename) != adapter.metadata[key]:
            raise ValueError(f"Adapter {filename} provenance mismatch")
    processor = json.loads((MODEL_DIR / "preprocessor_config.json").read_text())
    if processor.get("size") != {"height": 518, "width": 518}:
        raise ValueError("Selected adapter requires the original 518x518 processor")
    for path, digest in adapter.metadata["base_source_sha256"].items():
        source = Path(path)
        if source.name == "extract_frozen_features.py":
            target = Path(__file__).with_name(source.name)
        elif source.parent.name == "model":
            target = MODEL_DIR / source.name
        else:
            continue
        if sha256(target) != digest:
            raise ValueError(f"Frozen scorer implementation differs: {target}")


def predict_images(adapter, paths, root, dicom, model, processor, tokenizer, args, output):
    extractor = FrozenFeatureExtractor(model, tokenizer, adapter.prompts)
    images = TestImages(paths, root, processor, dicom)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                  drop_last=False, pin_memory=args.device.startswith("cuda"))
    if args.workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    baseline = np.empty((len(paths), len(adapter.class_names)), dtype=np.float64)
    adapted = np.empty_like(baseline)
    offset, api_check = 0, None
    with torch.inference_mode():
        for pixels, _, indices in DataLoader(images, **kwargs):
            if pixels.shape[-2:] != (518, 518):
                raise ValueError("Unexpected input size; expected original 518x518 preprocessing")
            stop = offset + len(pixels)
            if not np.array_equal(indices.numpy(), np.arange(offset, stop)):
                raise ValueError("Image order changed")
            pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
            values = extractor(pixels)
            if api_check is None:
                api_check = extractor.verify_official_api(pixels)
            base = values["base_logits"].cpu().numpy()
            local = values["class_local"].cpu().numpy()
            baseline[offset:stop] = base
            adapted[offset:stop] = adapter.predict(base, local)
            offset = stop
            write_json(output / "progress.json", {
                "status": "inference", "images_completed": offset, "images_total": len(paths),
                "updated_at": now()})
            if offset == len(pixels) or offset % 512 == 0 or offset == len(paths):
                print(f"{adapter.metadata['dataset']}: {offset}/{len(paths)}", flush=True)
    if offset != len(paths):
        raise ValueError("Incomplete image inference")
    np.save(output / "baseline_logits.npy", baseline, allow_pickle=False)
    np.save(output / "predictions.npy", adapted, allow_pickle=False)
    return baseline, adapted, api_check


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu or expose a free GPU")
    custom = bool(args.images)
    names = ["custom_images"] if custom else args.datasets
    model = processor = tokenizer = None
    results = {}
    try:
        for name in names:
            adapter_path = args.adapter or (args.adapter_root / name / "adapter.npz")
            adapter = RidgeAdapter.load(adapter_path)
            verify_base(adapter)
            spec = None
            if custom:
                paths = [str(path.expanduser().resolve()) for path in args.images]
                root, dicom = Path("/"), args.dicom
            else:
                if adapter.metadata["dataset"] != name:
                    raise ValueError("Selected adapter belongs to a different dataset")
                split_dir = args.split_dir or Path(adapter.metadata["split_dir"])
                spec = load_spec(split_dir, name, "test")
                expected_sha = next(value for path, value in adapter.metadata["split_sha256"].items()
                                    if Path(path).name == "test.csv")
                if sha256(split_dir / name / "test.csv") != expected_sha:
                    raise ValueError("Evaluation split differs from selected adapter provenance")
                if (tuple(spec["class_names"]) != adapter.class_names
                        or tuple(spec["prompts"]) != adapter.prompts):
                    raise ValueError("Dataset class/prompt order differs from adapter")
                paths = spec["image_paths"][:args.limit_test] if args.limit_test else spec["image_paths"]
                root, dicom = Path(spec["data_root"]), name in ("SIIM", "RSNA")
            if not paths:
                raise ValueError("No images")
            directory = args.output / name
            directory.mkdir()
            write_json(directory / "run_config.json", {
                "dataset": name, "adapter": str(adapter_path), "adapter_sha256": sha256(adapter_path),
                "adapter_provenance": adapter.metadata, "device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "batch_size": args.batch_size, "image_count": len(paths),
                "is_smoke": bool(args.limit_test and not custom),
                "method": "Frozen RadZero plus class-local weighted-ridge score residual",
                "map_interpretation": adapter.metadata["map_interpretation"],
                "source_sha256": {str(Path(__file__)): sha256(__file__),
                                  str(Path(__file__).with_name("ridge_adapter.py")):
                                  sha256(Path(__file__).with_name("ridge_adapter.py"))},
            })
            if model is None:
                model, processor, tokenizer = load_model(torch.device(args.device))
                model = model.float().requires_grad_(False).eval()
            baseline, adapted, api_check = predict_images(
                adapter, paths, root, dicom, model, processor, tokenizer, args, directory)
            with (directory / "predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image_path", *[f"base_logit:{c}" for c in adapter.class_names],
                                 *[f"adapted_logit:{c}" for c in adapter.class_names]])
                writer.writerows([path, *base.tolist(), *score.tolist()]
                                 for path, base, score in zip(paths, baseline, adapted))
            result = {"status": "complete", "image_count": len(paths),
                      "is_smoke": bool(args.limit_test and not custom),
                      "official_api_check": api_check}
            if spec is not None:
                score_spec = {**spec, "image_paths": paths, "labels": spec["labels"][:len(paths)]}
                result["baseline_metrics"] = compute_metrics(score_spec, baseline)
                result["metrics"] = compute_metrics(score_spec, adapted)
                result["metrics"]["score_type"] = "raw frozen RadZero logits plus weighted-ridge residual"
            write_json(directory / "result.json", result)
            write_json(directory / "progress.json", {"status": "complete", "updated_at": now()})
            results[name] = result
            write_json(args.output / "summary.json", {"status": "running", "datasets": results})
        write_json(args.output / "summary.json", {"status": "complete", "datasets": results})
    except BaseException as error:
        write_json(args.output / "failure.json", {
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--adapter", type=Path,
                        help="Explicit adapter for one --datasets entry, or for unlabelled --images")
    parser.add_argument("--images", type=Path, nargs="+", help="Unlabelled PNG/JPG/DICOM files")
    parser.add_argument("--dicom", action="store_true", help="Decode custom --images as DICOM")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.limit_test < 0:
        parser.error("Invalid batch/workers/limit")
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
