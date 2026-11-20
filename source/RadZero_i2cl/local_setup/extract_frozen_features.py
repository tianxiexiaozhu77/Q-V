#!/usr/bin/env python3
"""Cache compact, frozen RadZero features without exposing test labels to training.

Class-local features are the released cosine cross-attention's pooled vectors.
They do not permit recomputing attention after a query/alignment change. A later
score residual is not explained by the unchanged original similarity map.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from infer_batch import (
    ASSETS_DIR, MODEL_DIR, TestImages, checkpoint_identity, load_model, now,
    write_json,
)
from batch_data import DATASETS
from i2cl_data import load_spec

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


BASELINE_ROOT = ASSETS_DIR / "outputs/public5_posweighted_20261020_2014"
DEFAULT_SPLIT = ASSETS_DIR / "splits/native_public5_local_seed42"
FEATURE_NAMES = ("global_dino", "global_aligned", "class_local", "base_logits")
SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def public_spec(spec):
    """Keep image identifiers and prompts; deliberately discard label metadata."""
    return {
        key: spec[key]
        for key in ("name", "data_root", "image_paths", "ids", "groups",
                    "class_names", "prompts")
    }


def assert_disjoint(support, test):
    for key in ("image_paths", "ids", "groups"):
        if set(support[key]) & set(test[key]):
            raise ValueError(f"Support/test overlap in {key}")
    for key in ("class_names", "prompts"):
        if support[key] != test[key]:
            raise ValueError(f"Support/test {key} mismatch")


def feature_shapes(count, classes, hidden_dim=768):
    return {
        "global_dino": (count, 2, hidden_dim),
        "global_aligned": (count, 2, hidden_dim),
        "class_local": (count, classes, hidden_dim),
        "base_logits": (count, classes),
    }


def strict_baseline_check(actual, expected, atol=1e-6):
    if actual.shape != expected.shape:
        raise ValueError(f"Baseline shape changed: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite baseline logits")
    error = float(np.abs(actual.astype(np.float64) - expected).max(initial=0))
    if error > atol:
        raise ValueError(f"Historical baseline mismatch: max_abs_error={error:.9g} > {atol}")
    return {"passed": True, "max_abs_error": error, "atol": atol, "rtol": 0.0,
            "bitwise_equal": bool(np.array_equal(actual, expected))}


def read_baseline(path, paths, class_names):
    """Read only paths and logits; annotation columns are never consumed."""
    result = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        for index, row in enumerate(rows):
            if index >= len(paths) or int(row["index"]) != index:
                raise ValueError("Historical baseline index/count changed")
            if row["image_path"] != paths[index]:
                raise ValueError(f"Historical baseline image order changed at {index}")
            result.append([float(row[f"logit:{name}"]) for name in class_names])
    values = np.asarray(result, dtype=np.float32)
    if values.shape != (len(paths), len(class_names)) or not np.isfinite(values).all():
        raise ValueError("Incomplete or non-finite historical baseline")
    return values


def pool_original_matching(loss, text_features, vision):
    """Return exact official scores and its normalized pooled class features."""
    if loss.sim_op != "cos":
        raise ValueError("This cache schema requires the released cosine scorer")
    if loss.layer_norm is not None:
        vision = loss.layer_norm(vision)
    if not loss.use_vision_cls_token:
        vision = vision[:, 1:]
    batch, _, hidden = vision.shape
    classes = len(text_features)
    logits, attention = loss.compute_t2i_logits(
        text_features, vision, need_attn_weights=True
    )
    if attention is None or len(attention) != 1:
        raise ValueError("Expected the released single similarity-attention matrix")
    scores = attention[0]
    if scores.shape != (batch, classes, vision.shape[1]):
        raise ValueError(f"Unexpected attention shape {tuple(scores.shape)}")
    normalized_vision = F.normalize(vision, p=2, dim=-1)
    pooled = F.normalize(
        torch.matmul(F.softmax(scores, dim=-1), normalized_vision), p=2, dim=-1
    )
    logits = logits.reshape(classes, batch).T / loss.loss_temperature.exp()
    # Reconstruct independently from the cached vector, preserving every CLS
    # inclusion and temperature decision made by the original matching module.
    reconstructed = (
        F.normalize(text_features, p=2, dim=-1)[None] * pooled
    ).sum(dim=-1) / loss.loss_temperature.exp()
    torch.testing.assert_close(reconstructed, logits, atol=2e-5, rtol=1e-5)
    if pooled.shape != (batch, classes, hidden):
        raise ValueError("Unexpected pooled-feature shape")
    if not torch.isfinite(pooled).all() or not torch.isfinite(logits).all():
        raise ValueError("Non-finite pooled feature or original score")
    error = float((reconstructed - logits).abs().max())
    return pooled, logits, error


class FrozenFeatureExtractor(nn.Module):
    def __init__(self, model, tokenizer, prompts):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.loss = model.loss_fns["KeyPhraseAlignmentLoss"]
        self.prompts = tuple(prompts)
        device = next(model.parameters()).device
        encoded = tokenizer(
            list(prompts), padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()
                   if key in ("input_ids", "attention_mask")}
        self.encoded = encoded
        split = [{key: value[i:i + 1] for key, value in encoded.items()}
                 for i in range(len(prompts))]
        with torch.no_grad():
            text, _ = self.loss.compute_text_features(
                split, model.forward_text_model, ddp_gather=False
            )
        self.register_buffer("text_features", text.detach())
        self.reconstruction_max_abs_error = 0.0

    @torch.no_grad()
    def forward(self, pixels):
        self.model.eval()
        dino = self.model.vision_model(pixels)["last_hidden_state"]
        aligned = self.model.align_transformer(dino)
        pooled, logits, error = pool_original_matching(
            self.loss, self.text_features, aligned
        )
        self.reconstruction_max_abs_error = max(
            self.reconstruction_max_abs_error, error
        )
        values = {
            "global_dino": torch.stack((dino[:, 0], dino[:, 1:].mean(1)), dim=1),
            "global_aligned": torch.stack(
                (aligned[:, 0], aligned[:, 1:].mean(1)), dim=1
            ),
            "class_local": pooled,
            "base_logits": logits,
        }
        for name, value in values.items():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"Expected finite FP32 {name}")
        return values

    @torch.no_grad()
    def verify_official_api(self, pixels):
        checks = []
        for count in sorted({1, min(2, len(pixels))}):
            subset = pixels[:count]
            actual = self(subset)["base_logits"]
            expected = self.model.compute_logits(
                subset, [self.encoded]
            )["logits"].reshape(count, len(self.prompts))
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
            checks.append({"batch_size": count, "max_abs_error": float(
                (actual - expected).abs().max())})
        return {"passed": True, "checks": checks, "atol": 2e-5, "rtol": 1e-4}


def image_loader(spec, processor, args):
    images = TestImages(
        spec["image_paths"], Path(spec["data_root"]), processor,
        args.dataset in ("SIIM", "RSNA"),
    )
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  shuffle=False, drop_last=False, pin_memory=args.device.startswith("cuda"))
    if args.workers:
        kwargs.update(multiprocessing_context="spawn", persistent_workers=True,
                      prefetch_factor=2)
    return DataLoader(images, **kwargs)


def cache_split(name, spec, extractor, processor, args, expected=None):
    count, classes = len(spec["image_paths"]), len(spec["class_names"])
    shapes = feature_shapes(count, classes)
    arrays = {
        key: np.lib.format.open_memmap(
            args.output / f"{name}_{key}.npy", mode="w+", dtype=np.float32, shape=shape
        ) for key, shape in shapes.items()
    }
    for value in arrays.values():
        value.fill(np.nan)
    offset, last_print, timer = 0, 0, time.monotonic()
    baseline_check = {"passed": True, "checked_images": 0, "max_abs_error": 0.0,
                      "atol": 1e-6, "rtol": 0.0, "bitwise_equal": True}
    identity = None
    with torch.inference_mode():
        for pixels, _, indices in image_loader(spec, processor, args):
            stop = offset + len(pixels)
            if not np.array_equal(indices.numpy(), np.arange(offset, stop)):
                raise ValueError("Feature-cache sample order changed")
            pixels = pixels.to(args.device, dtype=torch.float32, non_blocking=True)
            values = extractor(pixels)
            if identity is None:
                identity = extractor.verify_official_api(pixels)
            for key, value in values.items():
                expected_shape = (len(pixels),) + shapes[key][1:]
                if tuple(value.shape) != expected_shape:
                    raise ValueError(f"{key} shape {tuple(value.shape)} != {expected_shape}")
                arrays[key][offset:stop] = value.cpu().numpy()
            if expected is not None:
                check = strict_baseline_check(
                    arrays["base_logits"][offset:stop], expected[offset:stop]
                )
                baseline_check["max_abs_error"] = max(
                    baseline_check["max_abs_error"], check["max_abs_error"]
                )
                baseline_check["bitwise_equal"] &= check["bitwise_equal"]
                baseline_check["checked_images"] = stop
            offset = stop
            progress = {
                "status": "extracting", "dataset": args.dataset, "split": name,
                "completed_images": offset, "total_images": count,
                "images_per_second": offset / max(time.monotonic() - timer, .001),
                "updated_at": now(),
            }
            write_json(args.output / "progress.json", progress)
            if offset == len(pixels) or offset - last_print >= 500 or offset == count:
                for array in arrays.values():
                    array.flush()
                print(f"{args.dataset} {name}: {offset}/{count} "
                      f"({progress['images_per_second']:.2f} images/s)", flush=True)
                last_print = offset
    if offset != count:
        raise ValueError(f"Incomplete {name} extraction: {offset}/{count}")
    files = {}
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite/incomplete {name} cache {key}")
        array.flush()
        path = args.output / f"{name}_{key}.npy"
        files[key] = {"file": path.name, "shape": list(array.shape),
                      "dtype": str(array.dtype), "sha256": sha256(path),
                      "size_bytes": path.stat().st_size}
    return {"count": count, "files": files, "official_api_check": identity,
            "historical_baseline_check": baseline_check if expected is not None else None}


def run(args):
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    support_raw = load_spec(args.split_dir, args.dataset, split="support")
    test_raw = load_spec(args.split_dir, args.dataset, split="test")
    assert_disjoint(support_raw, test_raw)
    support_labels = np.asarray(support_raw["labels"], dtype=np.int8).copy()
    support, test = public_spec(support_raw), public_spec(test_raw)
    split_fingerprint = support_raw["metadata"]["root_split_fingerprint"]
    support_sha = support_raw["metadata"]["support_sha256"]
    # No test labels survive beyond split validation, and they are not exported.
    del support_raw, test_raw
    baseline_dir = BASELINE_ROOT / args.dataset
    baseline_path = baseline_dir / "baseline_predictions.csv"
    baseline_config_path = baseline_dir / "run_config.json"
    baseline_config = (json.loads(baseline_config_path.read_text())
                       if baseline_config_path.is_file() else None)
    split_paths = [args.split_dir / "manifest.json",
                   args.split_dir / args.dataset / "manifest.json",
                   args.split_dir / args.dataset / "support.csv",
                   args.split_dir / args.dataset / "test.csv"]
    split_hashes = {str(path): sha256(path) for path in split_paths}
    baseline_matches_split = baseline_config is not None
    if baseline_matches_split:
        for name in ("support.csv", "test.csv"):
            current = split_hashes[str(args.split_dir / args.dataset / name)]
            previous = [digest for path, digest in baseline_config["split_sha256"].items()
                        if Path(path).name == name]
            baseline_matches_split &= previous == [current]
    if baseline_matches_split:
        expected = read_baseline(baseline_path, test["image_paths"], test["class_names"])
    elif args.allow_missing_baseline:
        # For a newly generated independent support split there is no archived
        # baseline CSV.  The frozen encoder still computes base logits below;
        # we simply omit the historical parity assertion and retain an audit
        # marker in the cache manifest.
        expected = None
    else:
        raise ValueError("Historical baseline is missing or uses a different split; "
                         "pass --allow-missing-baseline only for a new audited split")
    checkpoint = checkpoint_identity()
    if baseline_config is not None and checkpoint != baseline_config["checkpoint"]:
        raise ValueError("Historical baseline checkpoint differs")
    source_paths = [Path(__file__).resolve()] + [
        Path(__file__).with_name(name).resolve()
        for name in ("infer.py", "infer_batch.py", "i2cl_data.py", "batch_data.py",
                     "asset_paths.py")
    ] + sorted(MODEL_DIR.glob("*.py"))
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "loading_model",
        "dataset": args.dataset, "created_at": now(), "checkpoint": checkpoint,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "split_sha256": split_hashes, "root_split_fingerprint": split_fingerprint,
        "support_sha256": support_sha, "split_dir": str(args.split_dir),
        "support_identity": fingerprint(support), "test_identity": fingerprint(test),
        "class_names": support["class_names"], "prompts": support["prompts"],
        "precision": "float32; TF32 disabled", "batch_size": args.batch_size,
        "workers": args.workers, "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "preprocessor_sha256": sha256(MODEL_DIR / "preprocessor_config.json"),
        "baseline_reference": ({"path": str(baseline_path), "sha256": sha256(baseline_path),
                                "run_config_sha256": sha256(baseline_dir / "run_config.json")}
                               if expected is not None else None),
        "historical_baseline_check": ("passed" if expected is not None else
                                      "not_available_for_new_split; base logits computed by frozen encoder"),
        "test_labels_exported": False, "trainable_parameter_count": 0,
        "support_token_reuse": False,
        "global_feature_order": ["CLS", "mean_of_patch_tokens"],
        "limitations": [
            "Class-local vectors use the original fixed text query and attention.",
            "Compact features cannot recompute attention for changed query/alignment.",
            "Original similarity maps do not explain a subsequently added score residual.",
            "Test labels are not exported; independent evaluation must load the verified split.",
        ],
    }
    manifest["identity"] = fingerprint(manifest)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "progress.json", {
        "status": "loading_model", "dataset": args.dataset, "updated_at": now()
    })
    np.save(args.output / "support_labels.npy", support_labels, allow_pickle=False)
    write_json(args.output / "support_metadata.json", support)
    write_json(args.output / "test_metadata.json", test)
    model, processor, tokenizer = load_model(torch.device(args.device))
    model = model.float().requires_grad_(False).eval()
    extractor = FrozenFeatureExtractor(model, tokenizer, support["prompts"])
    if any(parameter.requires_grad for parameter in extractor.parameters()):
        raise RuntimeError("A supposedly frozen parameter requires gradients")
    np.save(args.output / "text_features.npy",
            extractor.text_features.detach().cpu().numpy(), allow_pickle=False)
    manifest["loss_settings"] = {
        "sim_op": extractor.loss.sim_op,
        "use_vision_cls_token": extractor.loss.use_vision_cls_token,
        "loss_temperature": float(extractor.loss.loss_temperature.exp()),
        "attention_temperature": float(
            extractor.loss.attn_temperature.exp()
            if extractor.loss.attn_temperature is not None
            else extractor.loss.loss_temperature.exp()
        ),
    }
    manifest["status"] = "extracting"
    write_json(args.output / "manifest.json", manifest)
    manifest["splits"] = {}
    for name, spec in (("support", support), ("test", test)):
        manifest["splits"][name] = cache_split(
            name, spec, extractor, processor, args, expected if name == "test" else None
        )
        write_json(args.output / "manifest.json", manifest)
    manifest["pooled_logit_reconstruction"] = {
        "passed": True, "max_abs_error": extractor.reconstruction_max_abs_error,
        "atol": 2e-5, "rtol": 1e-5,
    }
    manifest["auxiliary_files"] = {
        path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in (args.output / "support_labels.npy",
                     args.output / "text_features.npy",
                     args.output / "support_metadata.json",
                     args.output / "test_metadata.json")
    }
    manifest.update(status="complete", completed_at=now())
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "progress.json", {
        "status": "complete", "dataset": args.dataset, "updated_at": now(),
        "support_images": len(support["image_paths"]), "test_images": len(test["image_paths"]),
    })
    print(f"{args.dataset}: compact frozen feature cache complete: {args.output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--device", default="cuda")
    result.add_argument("--allow-missing-baseline", action="store_true",
                        help="Allow a newly generated split without an archived baseline parity CSV; base logits are still computed and recorded")
    return result


def main():
    args = parser().parse_args()
    args.output = args.output.expanduser().resolve()
    args.split_dir = args.split_dir.expanduser().resolve()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    # Exclusive creation intentionally refuses resumptions and accidental reuse.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        failure = {
            "status": "failed", "dataset": args.dataset, "updated_at": now(),
            "error": repr(error), "traceback": traceback.format_exc(),
        }
        write_json(args.output / "progress.json", failure)
        path = args.output / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(failure)
        write_json(path, manifest)
        raise


if __name__ == "__main__":
    main()
