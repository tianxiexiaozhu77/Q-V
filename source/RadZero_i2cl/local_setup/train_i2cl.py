#!/usr/bin/env python3
"""Calibrate eight frozen-RadZero I2CL coefficients and evaluate a held-out split.

This is a visual adaptation of I2CL, not an implementation of its LLM benchmark.
The same per-class support images build the fixed visual context and train the
injection coefficients.  The final fixed-epoch checkpoint is evaluated once;
evaluation labels never select coefficients, epochs, or hyperparameters.
Positive BCE terms use per-class negatives/positives counts from the complete
training support set, fixed across all minibatches.  An identity-centered
coefficient penalty discourages departures from alpha=0, beta=1.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from infer_batch import (
    ASSETS_DIR, MODEL_DIR, TestImages, checkpoint_identity, dataset_identity,
    load_model, now, write_json,
)
from batch_data import DATASETS, load_dataset
from batch_metrics import compute_metrics
from i2cl_data import load_spec
from i2cl_model import RadZeroI2CL

import numpy as np
import torch
from torch.utils.data import DataLoader


LOSS_DESCRIPTION = "BCEWithLogitsLoss with per-class support negatives/positives pos_weight"
OBJECTIVE_DESCRIPTION = "weighted BCE + lambda_reg * (mean(alpha^2) + mean((beta - 1)^2))"


def regularization_metadata(lambda_reg):
    if not math.isfinite(lambda_reg) or lambda_reg < 0:
        raise ValueError("lambda-reg must be finite and nonnegative")
    return {
        "type": "identity_coefficients_l2",
        "lambda": float(lambda_reg),
        "formula": "mean(alpha^2) + mean((beta - 1)^2)",
        "alpha_target": 0.0,
        "beta_target": 1.0,
    }


def coefficient_identity_penalty(coefficients):
    """Penalize departures from the original model, not beta's distance to zero."""
    alpha = coefficients[..., 0]
    beta = coefficients[..., 1]
    return alpha.square().mean() + (beta - 1).square().mean()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def tensor_digest(module):
    """Hash original tensor values, not serialization containers or GPU pointers."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def spec_identity(spec):
    return {
        "image_paths": list(spec["image_paths"]),
        "labels": np.asarray(spec["labels"], dtype=np.int8).tolist(),
        "ids": list(spec["ids"]),
        "groups": list(spec["groups"]),
        "class_names": list(spec["class_names"]),
        "prompts": list(spec["prompts"]),
    }


def limited_spec(spec, limit):
    if not limit or limit >= len(spec["image_paths"]):
        return spec
    result = dict(spec)
    for key in ("image_paths", "labels", "ids", "groups"):
        result[key] = spec[key][:limit]
    result["metadata"] = dict(spec.get("metadata", {}), limit_test=limit)
    return result


def image_loader(spec, processor, args, batch_size):
    root = Path(spec.get("data_root", ASSETS_DIR / "data"))
    dataset = TestImages(
        spec["image_paths"], root, processor, args.dataset in ("SIIM", "RSNA")
    )
    kwargs = {
        "batch_size": batch_size,
        "num_workers": args.workers,
        "shuffle": False,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.workers:
        kwargs.update(
            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2
        )
    return DataLoader(dataset, **kwargs)


def save_predictions(path, spec, logits):
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(spec["labels"], dtype=np.int8)
    if logits.shape != labels.shape or not np.isfinite(logits).all():
        raise RuntimeError("Prediction shape/finite-value check failed")
    # Avoid exp overflow even if an unconstrained trained coefficient is large.
    probabilities = torch.sigmoid(torch.from_numpy(logits).double()).numpy()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "image_path"]
            + [f"label:{name}" for name in spec["class_names"]]
            + [f"logit:{name}" for name in spec["class_names"]]
            + [f"score:{name}" for name in spec["class_names"]]
        )
        for index, image_path in enumerate(spec["image_paths"]):
            writer.writerow(
                [index, image_path] + labels[index].tolist()
                + logits[index].tolist() + probabilities[index].tolist()
            )
    temporary.replace(path)


def load_recorded_baseline(args, spec):
    """Re-index the verified original full-test logits onto this exact holdout."""
    directory = args.baseline_dir / args.dataset
    predictions = directory / "predictions.csv"
    if not predictions.exists():
        return None, {
            "mode": "computed_in_same_evaluation",
            "reason": "Original baseline predictions.csv is unavailable",
        }
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Existing baseline lacks a verifiable manifest")
    full_spec = load_dataset(args.dataset, ASSETS_DIR / "data")
    _, _, expected_fingerprint = dataset_identity(full_spec, 0)
    manifest = json.loads(manifest_path.read_text())
    if manifest["fingerprint"] != expected_fingerprint:
        raise RuntimeError("Saved baseline uses different checkpoint/data/prompts")
    with predictions.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    lookup = {row["image_path"]: row for row in records}
    if len(lookup) != len(records):
        raise RuntimeError("Original baseline contains duplicate image paths")
    if set(lookup) != set(full_spec["image_paths"]):
        raise RuntimeError("Original baseline does not cover its full source split")
    result = []
    for image_path, expected_labels in zip(spec["image_paths"], spec["labels"]):
        if image_path not in lookup:
            raise RuntimeError(f"Holdout image absent from original baseline: {image_path}")
        row = lookup[image_path]
        actual_labels = [int(row[f"label:{name}"]) for name in spec["class_names"]]
        if actual_labels != np.asarray(expected_labels).tolist():
            raise RuntimeError(f"Original baseline target mismatch: {image_path}")
        result.append([float(row[f"logit:{name}"]) for name in spec["class_names"]])
    logits = np.asarray(result, dtype=np.float32)
    if logits.shape != np.asarray(spec["labels"]).shape or not np.isfinite(logits).all():
        raise RuntimeError("Invalid re-indexed original baseline logits")
    return logits, {
        "mode": "original_predictions_reindexed_by_image_path_and_checked_labels",
        "predictions_path": str(predictions),
        "predictions_sha256": sha256(predictions),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "source_fingerprint": expected_fingerprint,
        "sample_count": len(logits),
        "comparison": "Identical held-out images, class order, and targets for both methods",
    }


def assert_split_contract(support, test):
    if support["class_names"] != test["class_names"] or support["prompts"] != test["prompts"]:
        raise RuntimeError("Support/test prompts or label order differ")
    if set(support["image_paths"]) & set(test["image_paths"]):
        raise RuntimeError("Support image leaked into evaluation")
    if set(support["ids"]) & set(test["ids"]):
        raise RuntimeError("Support ID leaked into evaluation")
    if set(support["groups"]) & set(test["groups"]):
        raise RuntimeError("Support patient/study/hash group leaked into evaluation")
    for spec in (support, test):
        labels = np.asarray(spec["labels"])
        if labels.ndim != 2 or len(spec["image_paths"]) != labels.shape[0]:
            raise RuntimeError("Invalid split shape")
        if not np.isin(labels, [0, 1]).all():
            raise RuntimeError("Expected binary multilabel targets")
        if len(set(spec["image_paths"])) != len(spec["image_paths"]):
            raise RuntimeError("Duplicate support/evaluation image paths")
        if not ((labels.sum(0) > 0) & (labels.sum(0) < len(labels))).all():
            raise RuntimeError("Full support and holdout must retain both states per class")


def extract_support(adapter, spec, processor, args, device):
    arrays = []
    first_pixels = None
    count = 0
    loader = image_loader(spec, processor, args, args.eval_batch_size)
    with torch.no_grad():
        for pixels, _, indices in loader:
            if not np.array_equal(indices.numpy(), np.arange(count, count + len(pixels))):
                raise RuntimeError("Support extraction changed sample order")
            pixels = pixels.to(device, non_blocking=True)
            if first_pixels is None:
                first_pixels = pixels[:2].clone()
            elif len(first_pixels) < 2:
                first_pixels = torch.cat([first_pixels, pixels[:2 - len(first_pixels)]])
            tokens = adapter.encode_images(pixels)
            if tokens.shape[1:] != (1370, 768) or not torch.isfinite(tokens).all():
                raise RuntimeError(f"Unexpected visual token shape/value: {tokens.shape}")
            arrays.append(tokens.detach().cpu())
            count += len(tokens)
            print(f"{args.dataset}: support tokens {count}/{len(spec['image_paths'])}", flush=True)
    del loader
    tokens = torch.cat(arrays)
    if len(tokens) != len(spec["image_paths"]):
        raise RuntimeError("Support cache does not contain exactly the selected images")
    token_path = args.output / "support_tokens.npy"
    temporary = token_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, tokens.numpy(), allow_pickle=False)
    temporary.replace(token_path)
    write_json(args.output / "support_cache.json", {
        "token_path": str(token_path),
        "token_sha256": sha256(token_path),
        "shape": list(tokens.shape),
        "dtype": "float32",
        "token_stage": "frozen DINO output BEFORE the alignment transformer",
        "support_identity": fingerprint(spec_identity(spec)),
        "context_support_identity": fingerprint(spec_identity(spec)),
        "calibration_support_identity": fingerprint(spec_identity(spec)),
    })
    return tokens.to(device), first_pixels


def verify_identity(adapter, model, tokenizer, pixels):
    previous = adapter.coefficients.detach().clone()
    checks = []
    adapter.eval()
    try:
        with torch.no_grad():
            adapter.coefficients[..., 0].zero_()
            adapter.coefficients[..., 1].fill_(1.0)
            for count in sorted({1, min(2, len(pixels))}):
                subset = pixels[:count]
                tokens = adapter.encode_images(subset)
                actual = adapter.forward_tokens(tokens, return_maps=True)
                tokenized = tokenizer(
                    list(adapter.prompts), padding=True, truncation=True, return_tensors="pt"
                ).to(model.device)
                reference = model.compute_logits(subset, [tokenized])
                expected_logits = reference["logits"].reshape_as(actual["logits"])
                expected_maps = reference["similarity_scores"].reshape_as(
                    actual["similarity_scores"]
                )
                torch.testing.assert_close(
                    actual["logits"], expected_logits, atol=2e-5, rtol=1e-4
                )
                torch.testing.assert_close(
                    actual["similarity_scores"], expected_maps, atol=2e-5, rtol=1e-4
                )
                checks.append({
                    "batch_size": count,
                    "max_logit_error": float((actual["logits"] - expected_logits).abs().max()),
                    "max_map_error": float(
                        (actual["similarity_scores"] - expected_maps).abs().max()
                    ),
                })
    finally:
        with torch.no_grad():
            adapter.coefficients.copy_(previous)
    return {"passed": True, "alpha": 0.0, "beta": 1.0, "checks": checks}


def make_support_criterion(targets):
    """Estimate fixed class weights from all support labels, never a minibatch."""
    if targets.ndim != 2 or targets.numel() == 0:
        raise ValueError("Support targets must be a nonempty [images, classes] tensor")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Support targets must contain only binary 0/1 labels")
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    if not ((positives > 0) & (negatives > 0)).all():
        raise ValueError("Every support class must have both positive and negative examples")
    # [C] broadcasts over [B, C], including single-class SIIM/RSNA.
    pos_weight = negatives / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    weighting = {
        "statistics_split": "support",
        "positive_counts": positives.detach().cpu().to(torch.int64).tolist(),
        "negative_counts": negatives.detach().cpu().to(torch.int64).tolist(),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "reduction": "mean",
    }
    return criterion, weighting


def evaluate_support_loss(adapter, tokens, targets, batch_size, criterion):
    adapter.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            stop = min(start + batch_size, len(tokens))
            logits = adapter.forward_tokens(tokens[start:stop])["logits"]
            batch_targets = targets[start:stop]
            loss = criterion(logits, batch_targets)
            total += float(loss) * batch_targets.numel()
    return total / targets.numel()


def calibrate(adapter, model, tokens, labels, args):
    trainable = [(name, value) for name, value in adapter.named_parameters() if value.requires_grad]
    if len(trainable) != 1 or trainable[0][1] is not adapter.coefficients:
        raise RuntimeError(f"Unexpected trainable parameters: {[name for name, _ in trainable]}")
    if adapter.coefficients.numel() != 8 or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("This protocol requires exactly eight trainable scalars and a frozen base")
    targets = torch.tensor(labels, device=tokens.device, dtype=torch.float32)
    criterion, loss_weighting = make_support_criterion(targets)
    regularization = regularization_metadata(args.lambda_reg)
    print(json.dumps({
        "stage": "training_loss_setup", "dataset": args.dataset,
        "loss": LOSS_DESCRIPTION, "objective": OBJECTIVE_DESCRIPTION,
        "regularization": regularization, **loss_weighting,
    }), flush=True)
    initial_coefficients = adapter.coefficients.detach().cpu().tolist()
    initial_loss = evaluate_support_loss(adapter, tokens, targets, args.grad_batch_size, criterion)
    initial_penalty = float(coefficient_identity_penalty(adapter.coefficients.detach()))
    context_hash = tensor_sha256(adapter.context_vectors)
    optimizer = torch.optim.AdamW(
        [adapter.coefficients], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    nonzero_gradient = torch.zeros_like(adapter.coefficients, dtype=torch.bool)
    first_gradient = None
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        order = torch.randperm(len(tokens), generator=generator)
        total_loss = 0.0
        total_bce = 0.0
        total_penalty = 0.0
        total_regularization = 0.0
        examples = 0
        for start in range(0, len(tokens), args.grad_batch_size):
            indices = order[start:start + args.grad_batch_size].to(tokens.device)
            optimizer.zero_grad(set_to_none=True)
            logits = adapter.forward_tokens(
                tokens.index_select(0, indices), training_noise=True
            )["logits"]
            loss_cls = criterion(logits, targets.index_select(0, indices))
            penalty = coefficient_identity_penalty(adapter.coefficients)
            loss_reg = args.lambda_reg * penalty
            # Zero explicitly recovers the previous weighted-BCE optimization.
            loss = loss_cls + loss_reg if args.lambda_reg else loss_cls
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite calibration loss at epoch {epoch}")
            loss.backward()
            gradient = adapter.coefficients.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Missing or non-finite injection-coefficient gradients")
            if first_gradient is None:
                first_gradient = gradient.detach().cpu().tolist()
            nonzero_gradient |= gradient != 0
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Frozen RadZero unexpectedly received a gradient")
            optimizer.step()
            if not torch.isfinite(adapter.coefficients).all():
                raise RuntimeError("Non-finite injection coefficients")
            total_loss += float(loss.detach()) * len(indices)
            total_bce += float(loss_cls.detach()) * len(indices)
            total_penalty += float(penalty.detach()) * len(indices)
            total_regularization += float(loss_reg.detach()) * len(indices)
            examples += len(indices)
        if examples != len(tokens):
            raise RuntimeError("Epoch did not consume exactly the fixed support set")
        entry = {
            "epoch": epoch,
            "train_bce_with_noise": total_bce / examples,
            "train_coefficient_penalty": total_penalty / examples,
            "train_regularization_loss": total_regularization / examples,
            "train_total_loss_with_noise": total_loss / examples,
            "examples_seen": examples,
            "optimizer_steps": math.ceil(len(tokens) / args.grad_batch_size),
            "coefficients": adapter.coefficients.detach().cpu().tolist(),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(entry)
        write_json(args.output / "train_history.json", history)
        write_json(args.output / "progress.json", {
            "status": "training", "dataset": args.dataset, "epoch": epoch,
            "epochs": args.epochs, "support_count": len(tokens), "updated_at": now(),
            "train_bce_with_noise": entry["train_bce_with_noise"],
            "train_coefficient_penalty": entry["train_coefficient_penalty"],
            "train_regularization_loss": entry["train_regularization_loss"],
            "train_total_loss_with_noise": entry["train_total_loss_with_noise"],
        })
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"{args.dataset}: epoch {epoch}/{args.epochs}, "
                f"support pos-weighted BCE={entry['train_bce_with_noise']:.6f}, "
                f"reg={entry['train_regularization_loss']:.6f}, "
                f"total={entry['train_total_loss_with_noise']:.6f}, "
                f"{entry['elapsed_seconds']:.1f}s", flush=True,
            )
    if not bool(nonzero_gradient.all()):
        raise RuntimeError("Some injection coefficients never received a nonzero gradient")
    if tensor_sha256(adapter.context_vectors) != context_hash:
        raise RuntimeError("Fixed context vectors changed during calibration")
    final_loss = evaluate_support_loss(adapter, tokens, targets, args.grad_batch_size, criterion)
    final_penalty = float(coefficient_identity_penalty(adapter.coefficients.detach()))
    return {
        "loss_weighting": loss_weighting,
        "regularization": regularization,
        "initial_support_bce_without_noise": initial_loss,
        "final_support_bce_without_noise": final_loss,
        "initial_coefficient_penalty": initial_penalty,
        "final_coefficient_penalty": final_penalty,
        "initial_regularization_loss": args.lambda_reg * initial_penalty,
        "final_regularization_loss": args.lambda_reg * final_penalty,
        "initial_support_total_loss_without_noise": initial_loss + args.lambda_reg * initial_penalty,
        "final_support_total_loss_without_noise": final_loss + args.lambda_reg * final_penalty,
        "initial_coefficients": initial_coefficients,
        "final_coefficients": adapter.coefficients.detach().cpu().tolist(),
        "first_gradient": first_gradient,
        "all_eight_coefficients_received_nonzero_finite_gradients": True,
        "base_parameters_received_gradients": False,
        "context_unchanged": True,
        "context_sha256": context_hash,
        "epochs": args.epochs,
        "examples_per_epoch": len(tokens),
        "optimizer_steps": args.epochs * math.ceil(len(tokens) / args.grad_batch_size),
        "checkpoint_selection": "final fixed epoch; no validation/test selection",
        "elapsed_seconds": time.monotonic() - started,
    }


def checkpoint_roundtrip(adapter, checkpoint_path, tokens):
    adapter.eval()
    with torch.no_grad():
        reference = adapter.forward_tokens(tokens[:2], return_maps=True)
        reference = {key: value.detach().clone() for key, value in reference.items()}
        adapter.coefficients.zero_()
        adapter.context_vectors.zero_()
        restored = torch.load(checkpoint_path, map_location="cpu")
        adapter.load_adaptation_state(restored["adaptation_state"])
        actual = adapter.forward_tokens(tokens[:2], return_maps=True)
        errors = {}
        for key in ("logits", "similarity_scores"):
            torch.testing.assert_close(actual[key], reference[key], atol=1e-6, rtol=1e-6)
            errors[key] = float((actual[key] - reference[key]).abs().max())
    return {
        "passed": True,
        "method": "zero coefficients/context in memory, then reload serialized checkpoint",
        "max_absolute_errors": errors,
    }


def evaluate(adapter, spec, processor, baseline_logits, args, device):
    predictions = []
    computed_baseline = []
    offset = 0
    last_logged = 0
    started = time.monotonic()
    baseline_check = None
    adapter.eval()
    loader = image_loader(spec, processor, args, args.eval_batch_size)
    with torch.no_grad():
        for pixels, _, indices in loader:
            if not np.array_equal(indices.numpy(), np.arange(offset, offset + len(pixels))):
                raise RuntimeError("Evaluation row order changed")
            tokens = adapter.encode_images(pixels.to(device, non_blocking=True))
            logits = adapter.forward_tokens(tokens)["logits"]
            predictions.append(logits.cpu().numpy())
            if baseline_logits is None or baseline_check is None:
                original = adapter.forward_tokens(tokens, inject=False)["logits"]
                if baseline_logits is None:
                    computed_baseline.append(original.cpu().numpy())
                else:
                    recorded = torch.tensor(
                        baseline_logits[offset:offset + len(tokens)], device=device
                    )
                    torch.testing.assert_close(original, recorded, atol=2e-4, rtol=1e-4)
                    baseline_check = {
                        "passed": True, "checked_images": len(tokens),
                        "max_logit_error": float((original - recorded).abs().max()),
                        "atol": 2e-4, "rtol": 1e-4,
                    }
            offset += len(tokens)
            elapsed = time.monotonic() - started
            progress = {
                "status": "evaluating", "dataset": args.dataset,
                "completed_images": offset, "total_images": len(spec["image_paths"]),
                "images_per_second": offset / max(elapsed, 0.001),
                "elapsed_seconds": elapsed, "updated_at": now(),
            }
            write_json(args.output / "progress.json", progress)
            if offset == len(tokens) or offset - last_logged >= 500 or offset == len(spec["image_paths"]):
                print(
                    f"{args.dataset}: evaluation {offset}/{len(spec['image_paths'])} "
                    f"({progress['images_per_second']:.2f} images/s)", flush=True,
                )
                last_logged = offset
    del loader
    if offset != len(spec["image_paths"]):
        raise RuntimeError("Incomplete evaluation")
    if baseline_logits is None:
        baseline_logits = np.concatenate(computed_baseline)
        baseline_check = {
            "passed": True, "checked_images": offset,
            "method": "Base logits recomputed on every held-out image with injection disabled",
        }
    return np.concatenate(predictions), baseline_logits, baseline_check


def metric_deltas(original, adapted):
    keys = ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")
    return {
        key: (
            None if original[key] is None or adapted[key] is None
            else 100.0 * (adapted[key] - original[key])
        )
        for key in keys
    }


def run(args):
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but inaccessible; run with GPU access")
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    support = load_spec(args.split_dir, args.dataset, split="support")
    full_test = load_spec(args.split_dir, args.dataset, split="test")
    assert_split_contract(support, full_test)
    _, loss_weighting = make_support_criterion(
        torch.as_tensor(support["labels"], dtype=torch.float32)
    )
    test = limited_spec(full_test, args.limit_test)
    baseline_logits, baseline_identity = load_recorded_baseline(args, test)
    selected_support = spec_identity(support)
    support_identity = fingerprint(selected_support)
    split_files = sorted(
        path for path in (args.split_dir / args.dataset).iterdir()
        if path.is_file() and path.suffix in (".csv", ".json")
    )
    sources = (
        "train_i2cl.py", "i2cl_model.py", "i2cl_data.py", "infer_batch.py",
        "infer.py", "batch_data.py", "batch_metrics.py",
    )
    config = {
        "protocol": "radzero-visual-i2cl-alignment-cls-shot5-v1",
        "dataset": args.dataset,
        "created_at": now(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "checkpoint": checkpoint_identity(),
        "source_sha256": {
            str(Path(__file__).with_name(name)): sha256(Path(__file__).with_name(name))
            for name in sources
        },
        "split_sha256": {str(path): sha256(path) for path in split_files},
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "preprocessor_sha256": sha256(MODEL_DIR / "preprocessor_config.json"),
        "support_identity": support_identity,
        "context_support_identity": support_identity,
        "calibration_support_identity": support_identity,
        "test_identity": fingerprint(spec_identity(test)),
        "support_count": len(support["image_paths"]),
        "full_holdout_count": len(full_test["image_paths"]),
        "evaluated_count": len(test["image_paths"]),
        "is_smoke_test": bool(args.limit_test),
        "class_names": support["class_names"],
        "prompts": support["prompts"],
        "split_metadata": support.get("metadata", {}),
        "baseline": baseline_identity,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "float32; TF32 disabled",
        "resolution": [518, 518],
        "trainable_scalars": 8,
        "sampling": "Each epoch shuffles and consumes the same fixed support set exactly once",
        "loss": LOSS_DESCRIPTION,
        "loss_weighting": loss_weighting,
        "objective": OBJECTIVE_DESCRIPTION,
        "regularization": regularization_metadata(args.lambda_reg),
        "checkpoint_selection": "final epoch fixed before testing",
        "score_description": "Sigmoid matching scores are not calibrated clinical probabilities",
    }
    config["fingerprint"] = fingerprint(config)
    write_json(args.output / "run_config.json", config)
    write_json(args.output / "support_identity.json", dict(
        selected_support,
        support_identity=support_identity,
        context_support_identity=support_identity,
        calibration_support_identity=support_identity,
    ))
    write_json(args.output / "progress.json", {
        "status": "loading_model", "dataset": args.dataset, "updated_at": now()
    })
    model, processor, tokenizer = load_model(device)
    adapter = RadZeroI2CL(model, tokenizer, support["prompts"], noise_scale=args.noise_scale)
    frozen_before = tensor_digest(model)
    tokens, first_pixels = extract_support(adapter, support, processor, args, device)
    adapter.build_context(tokens, batch_size=args.grad_batch_size)
    reference = verify_identity(adapter, model, tokenizer, first_pixels)
    write_json(args.output / "reference_equivalence.json", reference)
    training = calibrate(adapter, model, tokens, support["labels"], args)
    if training["loss_weighting"] != config["loss_weighting"]:
        raise RuntimeError("Training loss weights differ from recorded support statistics")
    if training["regularization"] != config["regularization"]:
        raise RuntimeError("Training regularization differs from recorded configuration")
    frozen_after = tensor_digest(model)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen RadZero tensors changed during calibration")
    frozen_check = {
        "passed": True, "before_sha256": frozen_before, "after_sha256": frozen_after,
        "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": adapter.coefficients.numel(),
    }
    write_json(args.output / "frozen_model_check.json", frozen_check)
    checkpoint = {
        "format": "radzero-visual-i2cl-v1",
        "adaptation_state": adapter.adaptation_state(),
        "run_fingerprint": config["fingerprint"],
        "dataset": args.dataset,
        "split_dir": str(args.split_dir),
        "model_identity": config["checkpoint"],
        "support_sha256": support["metadata"]["support_sha256"],
        "class_names": support["class_names"],
        "prompts": support["prompts"],
        "support_identity": support_identity,
        "context_support_identity": support_identity,
        "calibration_support_identity": support_identity,
        "run_config": config,
        "config": config,
        "training": training,
        "frozen_model_check": frozen_check,
    }
    checkpoint_path = args.output / "i2cl.pt"
    atomic_save(checkpoint_path, checkpoint)
    roundtrip = checkpoint_roundtrip(adapter, checkpoint_path, tokens)
    write_json(args.output / "checkpoint_reload_check.json", roundtrip)
    if tensor_sha256(adapter.context_vectors) != training["context_sha256"]:
        raise RuntimeError("Reloaded context differs from its trained fixed context")
    del tokens, first_pixels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logits, baseline_logits, baseline_check = evaluate(
        adapter, test, processor, baseline_logits, args, device
    )
    predictions_path = args.output / "predictions.csv"
    baseline_path = args.output / "baseline_predictions.csv"
    save_predictions(predictions_path, test, logits)
    save_predictions(baseline_path, test, baseline_logits)
    metrics = compute_metrics(test, logits)
    baseline_metrics = compute_metrics(test, baseline_logits)
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "baseline_metrics.json", baseline_metrics)
    write_json(args.output / "baseline_verification.json", baseline_check)
    result = {
        "status": "complete",
        "dataset": args.dataset,
        "is_smoke_test": bool(args.limit_test),
        "run_fingerprint": config["fingerprint"],
        "support_count": len(support["image_paths"]),
        "evaluated_count": len(test["image_paths"]),
        "sample_count": len(test["image_paths"]),
        "full_holdout_count": len(full_test["image_paths"]),
        "support_identity": support_identity,
        "context_support_identity": support_identity,
        "calibration_support_identity": support_identity,
        "trainable_scalars": 8,
        "resolution": [518, 518],
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "delta_pp": metric_deltas(baseline_metrics, metrics),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": training,
        "checks": {
            "reference_equivalence": reference,
            "base_unchanged": frozen_check,
            "checkpoint_reload": roundtrip,
            "recorded_baseline": baseline_check,
            "support_test_image_id_group_disjoint": True,
            "context_and_calibration_use_identical_support": True,
            "identity": reference,
            "gradient": {
                "passed": training["all_eight_coefficients_received_nonzero_finite_gradients"],
                "base_parameters_received_gradients": False,
                "first_gradient": training["first_gradient"],
            },
            "frozen": frozen_check,
            "context": {
                "passed": True,
                "sha256": training["context_sha256"],
                "context_support_identity": support_identity,
                "calibration_support_identity": support_identity,
            },
            "reload": roundtrip,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "baseline_predictions": str(baseline_path),
        "baseline_predictions_sha256": sha256(baseline_path),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": now(),
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {
        "status": "complete", "dataset": args.dataset,
        "completed_images": len(test["image_paths"]),
        "total_images": len(test["image_paths"]), "updated_at": now(),
        "is_smoke_test": bool(args.limit_test), "result": str(args.output / "result.json"),
    })
    print(
        f"COMPLETE {args.dataset}: support={len(support['image_paths'])}, "
        f"test={len(test['image_paths'])}, macro AUROC={metrics['macro_auroc']}, "
        f"macro PR AUC={metrics['macro_auprc']}", flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument(
        "--lambda-reg", type=float, default=0.01,
        help="Identity coefficient penalty strength (default: 0.01; 0 disables it)",
    )
    parser.add_argument("--noise-scale", type=float, default=0.001)
    parser.add_argument("--limit-test", type=int, default=0,
                        help="0 evaluates the full holdout; positive values mark a smoke test")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--baseline-dir", type=Path,
                        default=ASSETS_DIR / "outputs/public5_gpu0")
    args = parser.parse_args()
    if min(args.epochs, args.grad_batch_size, args.eval_batch_size) < 1:
        parser.error("epochs and batch sizes must be positive")
    if args.workers < 0 or args.limit_test < 0:
        parser.error("workers and limit-test must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.noise_scale < 0:
        parser.error("learning-rate must be positive; decay/noise must be nonnegative")
    try:
        regularization_metadata(args.lambda_reg)
    except ValueError as exc:
        parser.error(str(exc))
    for key in ("output", "split_dir", "baseline_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main():
    args = parse_args()
    outputs_root = (ASSETS_DIR / "outputs").resolve()
    if outputs_root not in args.output.parents:
        raise RuntimeError("Use a new experiment directory inside " + str(outputs_root))
    if args.output.exists():
        existing = [
            path for path in args.output.iterdir()
            if path.name not in ("worker.log", ".worker.lock")
        ]
        if existing:
            raise RuntimeError(
                f"Output is not a fresh worker directory: {args.output}; "
                "existing experiments are never overwritten"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns this output: {args.output}") from exc
        occupied = [
            "run_config.json", "progress.json", "result.json", "i2cl.pt",
            "train_history.json", "support_tokens.npy", "failure.json",
        ]
        if any((args.output / name).exists() for name in occupied):
            raise RuntimeError(f"Output already contains a run; select a new directory: {args.output}")
        try:
            run(args)
        except Exception:
            error = traceback.format_exc()
            write_json(args.output / "failure.json", {
                "status": "failed", "dataset": args.dataset,
                "failed_at": now(), "traceback": error,
            })
            write_json(args.output / "progress.json", {
                "status": "failed", "dataset": args.dataset, "updated_at": now(),
                "error": error.splitlines()[-1],
            })
            raise


if __name__ == "__main__":
    main()
