#!/usr/bin/env python3
r"""Compare original RadZero and a trained Node21 adapter on selected MHA images.

Examples (select the physical GPU outside this script):
    CUDA_VISIBLE_DEVICES=0 python local_setup/infer_node21_adapter.py \
        --checkpoint /path/to/best_adapter.pt --indices 0 1 2 --save-maps
    CUDA_VISIBLE_DEVICES=1 python local_setup/infer_node21_adapter.py \
        --checkpoint /path/to/best_adapter.pt --images /path/a.mha /path/b.mha
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from batch_metrics import compute_metrics
from infer import select_device
from infer_batch import (
    ASSETS_DIR,
    MODEL_DIR,
    checkpoint_identity,
    load_model,
    now,
    write_json,
)
from node21_adapter_data import DEFAULT_OUTPUT as DEFAULT_SPLITS, load_split
from node21_data import Node21Images, PREPROCESSING_PROTOCOL
from siim_adapter_model import RadZeroAdapterHead, context_from_model
import numpy as np
import torch
from torch.utils.data import DataLoader


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_adapter_checkpoint(path):
    """Read only the tensor/primitive payload written by train_node21_adapter.py."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "adapter_state_dict", "architecture", "prompt", "checkpoint",
        "epoch", "validation_metrics", "run_fingerprint",
    }
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError("Expected a best_adapter.pt produced by train_node21_adapter.py")
    architecture = saved["architecture"]
    if not isinstance(architecture, dict) or architecture != {
        "hidden_dim": 768, "bottleneck": 32, "scale": 0.1,
    }:
        raise ValueError(f"Unsupported adapter architecture: {architecture!r}")
    if saved["prompt"] != "There is Lung Nodule":
        raise ValueError("Checkpoint does not use the fixed Node21 training prompt")
    if not isinstance(saved["epoch"], int) or saved["epoch"] < 1:
        raise ValueError("Checkpoint must contain a trained epoch")
    if not isinstance(saved["adapter_state_dict"], dict):
        raise ValueError("Missing adapter tensor state")
    if any(
        not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
        for value in saved["adapter_state_dict"].values()
    ):
        raise ValueError("Adapter state contains non-tensor or non-finite values")
    return saved


def select_images(args):
    """Resolve an explicit image list or indexed rows of the custom Node21 split."""
    if args.images:
        paths = [path.expanduser().resolve() for path in args.images]
        rows = [
            {
                "index": index,
                "split_index": None,
                "image_path": str(path),
                "absolute_image_path": str(path),
                "label": None,
            }
            for index, path in enumerate(paths)
        ]
        selection = {"kind": "explicit_images", "split": None}
    else:
        spec = load_split(args.splits_dir, args.split)
        selected = args.indices if args.indices is not None else [0]
        if len(selected) != len(set(selected)):
            raise ValueError("Image indices must not repeat")
        if any(index < 0 or index >= len(spec["image_paths"]) for index in selected):
            raise ValueError(f"Indices must be in 0..{len(spec['image_paths']) - 1}")
        paths = [Path(spec["image_paths"][index]).resolve() for index in selected]
        rows = [
            {
                "index": position,
                "split_index": index,
                "image_path": spec["image_paths"][index],
                "absolute_image_path": str(paths[position]),
                "label": int(spec["labels"][index, 0]),
            }
            for position, index in enumerate(selected)
        ]
        selection = {
            "kind": "split_indices", "split": args.split, "indices": selected,
            "splits_dir": str(args.splits_dir),
            "split_fingerprint": spec["metadata"]["fingerprint"],
        }
    if len(paths) != len(set(paths)):
        raise ValueError("Input images must not repeat")
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".mha":
            raise ValueError(f"Expected an existing Node21 MHA (.mha): {path}")
    return paths, rows, selection


def write_predictions(path, rows):
    fields = [
        "index", "split_index", "image_path", "absolute_image_path", "label",
        "baseline_logit", "adapter_logit", "baseline_matching_score",
        "adapter_matching_score", "delta_logit",
    ]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def optional_metrics(rows):
    if any(row["label"] is None for row in rows):
        return None
    spec = {
        "name": "Node21", "class_names": ["Nodule"],
        "image_paths": [row["image_path"] for row in rows],
        "labels": np.asarray([[row["label"]] for row in rows], dtype=np.int8),
    }
    return {
        kind: compute_metrics(
            spec, np.asarray([[row[kind + "_logit"]] for row in rows], dtype=np.float64)
        )
        for kind in ("baseline", "adapter")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained best_adapter.pt; original RadZero weights are loaded separately")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--images", type=Path, nargs="+", help="One or more Node21 MHA files")
    inputs.add_argument("--indices", type=int, nargs="+",
                        help="Zero-based indices in --split; default: 0")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-maps", action="store_true",
                        help="Save original and adapted raw 37x37 spatial scores as NPZ batches")
    parser.add_argument("--output", type=Path,
                        help="Fresh output directory; default: timestamped directory under assets/outputs")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative")
    if args.images and args.split != "test":
        parser.error("--split applies to --indices, not explicit --images")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.splits_dir = args.splits_dir.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = ASSETS_DIR / "outputs" / "node21_adapter_inference" / stamp
    args.output = args.output.expanduser().resolve()
    if args.output == MODEL_DIR or MODEL_DIR in args.output.parents:
        parser.error("Inference outputs must be outside the original model snapshot")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; choose another --output")

    saved = read_adapter_checkpoint(args.checkpoint)
    identity = checkpoint_identity()
    if saved["checkpoint"] != identity:
        raise RuntimeError("Adapter was trained with a different original RadZero checkpoint")
    paths, rows, selection = select_images(args)
    device = select_device(args.device, torch)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    started_at = now()
    model, processor, tokenizer = load_model(device)
    model.requires_grad_(False).eval()
    loss_module, text_features = context_from_model(model, tokenizer, saved["prompt"])
    options = {"bottleneck": saved["architecture"]["bottleneck"],
               "scale": saved["architecture"]["scale"]}
    baseline = RadZeroAdapterHead(loss_module, text_features, **options).to(device)
    adapted = RadZeroAdapterHead(loss_module, text_features, **options).to(device)
    adapted.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
    baseline.requires_grad_(False).eval()
    adapted.requires_grad_(False).eval()
    images = Node21Images([str(path) for path in paths], processor)
    loader_args = {
        "batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers:
        loader_args.update(multiprocessing_context="spawn", persistent_workers=True)
    loader = DataLoader(images, **loader_args)
    args.output.mkdir(parents=True, exist_ok=True)
    maps_directory = args.output / "maps"
    if args.save_maps:
        maps_directory.mkdir()
    map_files = []
    completed = []
    with torch.inference_mode():
        for pixels, image_hw, indices in loader:
            tokens = model.forward_vision_model(pixels.to(device, non_blocking=True))["vision_tokens"]
            if tokens.shape[1:] != (1370, 768) or not torch.isfinite(tokens).all():
                raise RuntimeError("Invalid frozen image features")
            original = baseline(tokens, return_maps=args.save_maps)
            trained = adapted(tokens, return_maps=args.save_maps)
            for output in (original, trained):
                if output["logits"].shape != (len(indices), 1):
                    raise RuntimeError("Unexpected prediction shape")
                if any(not torch.isfinite(value).all() for value in output.values()):
                    raise RuntimeError("Non-finite prediction or similarity map")
                if args.save_maps and output["similarity_scores"].shape != (len(indices), 1, 37, 37):
                    raise RuntimeError("Unexpected spatial map shape")
            original_logits = original["logits"].cpu().flatten().tolist()
            trained_logits = trained["logits"].cpu().flatten().tolist()
            original_scores = original["logits"].sigmoid().cpu().flatten().tolist()
            trained_scores = trained["logits"].sigmoid().cpu().flatten().tolist()
            for position, index in enumerate(indices.tolist()):
                rows[index].update(
                    baseline_logit=original_logits[position],
                    adapter_logit=trained_logits[position],
                    baseline_matching_score=original_scores[position],
                    adapter_matching_score=trained_scores[position],
                    delta_logit=trained_logits[position] - original_logits[position],
                )
                completed.append(index)
            if args.save_maps:
                path = maps_directory / f"batch_{int(indices[0]):06d}_{int(indices[-1])+1:06d}.npz"
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream, indices=indices.numpy(), image_hw=image_hw.numpy(),
                        baseline_similarity_scores=original["similarity_scores"].cpu().numpy(),
                        adapter_similarity_scores=trained["similarity_scores"].cpu().numpy(),
                    )
                temporary.replace(path)
                map_files.append(str(path))
            if len(completed) == len(indices) or len(completed) == len(rows):
                print(f"Compared {len(completed)}/{len(rows)} Node21 images", flush=True)
    if completed != list(range(len(rows))):
        raise RuntimeError("Inference omitted or duplicated image rows")
    write_predictions(args.output / "predictions.csv", rows)
    result = {
        "status": "complete", "started_at": started_at, "completed_at": now(),
        "elapsed_seconds": time.monotonic() - started, "images": len(rows),
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection": selection, "prompt": saved["prompt"],
        "checkpoint": identity, "adapter_checkpoint": str(args.checkpoint),
        "adapter_checkpoint_sha256": sha256(args.checkpoint),
        "adapter_epoch": saved["epoch"], "training_run_fingerprint": saved["run_fingerprint"],
        "adapter_architecture": saved["architecture"],
        "adapter_parameters": sum(p.numel() for p in adapted.adapter.parameters()),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters())
                                 and all(not p.requires_grad for p in adapted.parameters()),
        "preprocessing": PREPROCESSING_PROTOCOL,
        "precision": "FP32; TF32 disabled",
        "metrics_on_selected_images": optional_metrics(rows),
        "predictions": rows,
        "outputs": {"csv": str(args.output / "predictions.csv"), "maps": map_files},
        "map_format": "Raw pre-sigmoid [N,1,37,37] spatial scores. To render at original "
                      "image size, bilinearly interpolate with align_corners=False, then sigmoid.",
        "scope": "Node21 supervised image-level nodule adapter inference on a custom public-data "
                 "split, not the hidden challenge test or nodule detection/FROC. Scores are "
                 "image-text matching scores; selection metrics describe only supplied images.",
    }
    write_json(args.output / "result.json", result)
    print(json.dumps({
        "status": result["status"], "images": result["images"],
        "adapter_epoch": result["adapter_epoch"], "output": str(args.output),
        "metrics_on_selected_images": result["metrics_on_selected_images"],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
