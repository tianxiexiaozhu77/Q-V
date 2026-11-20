#!/usr/bin/env python3
"""Explicit conservative I2CL visual adaptations, sharing each test DINO pass.

The six predeclared trials preserve support images, frozen RadZero, weighted
BCE, and fixed visual contexts. Identity initialization is OUR ablation:
the original I2CL config initializes [alpha,beta]=[0.1,1.0].
The schedule below implements the author's exact 5%-warmup/cosine expression,
not the alternative cosine restarted after warmup.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import train_i2cl as original_training
from i2cl_model import RadZeroI2CL
from infer_batch import load_model, checkpoint_identity, MODEL_DIR, ASSETS_DIR
from overnight_cache_methods import (
    DeadlineReached, check_deadline, parse_deadline, sha256, write_json,
    evaluate_test_only,
)
from overnight_linear_residual import FrozenCache

VERSION = "radzero-conservative-i2cl-explicit-active-parameters-v1"
AUTHOR_WRAPPER = Path("/storage/3090group2/zjx/code/medical/I2CL/wrapper.py")
AUTHOR_CONFIG = AUTHOR_WRAPPER.parent / "configs/config_i2cl.py"

VARIANTS = {
    "A": {"sites": "all", "learn_beta": True, "learning_rate": .003, "lambda_reg": .01},
    "B": {"sites": "all", "learn_beta": True, "learning_rate": .001, "lambda_reg": .01},
    "C": {"sites": "all", "learn_beta": False, "learning_rate": .003, "lambda_reg": .01},
    "D": {"sites": "all", "learn_beta": False, "learning_rate": .003, "lambda_reg": .1},
    "E": {"sites": "last_block", "learn_beta": True, "learning_rate": .003, "lambda_reg": .01},
    "F": {"sites": "last_block", "learn_beta": False, "learning_rate": .003, "lambda_reg": .01},
}


def author_lr_factor(step, total_steps, warmup_steps):
    if total_steps < 1 or not 1 <= warmup_steps <= total_steps or step < 0:
        raise ValueError("Invalid author-schedule bounds")
    if step <= warmup_steps:
        return step / warmup_steps
    return .5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))


def schedule_description(total_steps, epochs, examples, batch_size):
    warmup = max(1, int(.05 * total_steps))
    return {
        "name": "author_formula_5pct_warmup_cosine",
        "formula": "step/W if step<=W else (1+cos(pi*step/S))/2",
        "total_steps": total_steps, "warmup_steps": warmup,
        "warmup_fraction": .05,
        "first_optimizer_update_lr_multiplier": 0.,
        "order": "optimizer.step(); scheduler.step()",
        "batch_count": "ceil(N/batch); no support images dropped",
        "matches_author_floor_batch_count": examples % batch_size == 0,
        "epochs": epochs, "source": str(AUTHOR_WRAPPER),
        "source_lines": "285-291,339-340",
        "small_step_guard": "W=max(1,floor(0.05*S)); current full runs need no guard",
    }


class ConservativeI2CL(RadZeroI2CL):
    """An explicit subclass with only the active deltas in the optimizer."""

    def __init__(self, model, tokenizer, prompts, *, sites="all", learn_beta=True,
                 noise_scale=.001):
        super().__init__(model, tokenizer, prompts, noise_scale=noise_scale)
        if sites not in ("all", "last_block"):
            raise ValueError("sites must be all or last_block")
        self.variant_sites = sites
        self.learn_beta = bool(learn_beta)
        # The inherited compatibility tensor is frozen and always identity.
        # It is NOT optimized, and effective_coefficients is used by our hook.
        with torch.no_grad():
            self.coefficients[..., 0].zero_()
            self.coefficients[..., 1].fill_(1.)
        self.coefficients.requires_grad_(False)
        coordinates = []
        for layer in range(self.layer_count):
            if sites == "last_block" and layer != self.layer_count - 1:
                continue
            for module in range(len(self.MODULE_NAMES)):
                coordinates.append((layer, module, 0))
                if self.learn_beta:
                    coordinates.append((layer, module, 1))
        self.active_coordinates = tuple(coordinates)
        flat_indices = [(layer * 2 + module) * 2 + which
                        for layer, module, which in coordinates]
        self.register_buffer("active_flat_indices",
                             torch.tensor(flat_indices, dtype=torch.long, device=self.device))
        self.active_deltas = nn.Parameter(torch.zeros(
            len(coordinates), device=self.device, dtype=self.dtype))
        self.active_sites = frozenset((layer, module) for layer, module, _ in coordinates)

    def effective_coefficients(self):
        identity = self.coefficients.reshape(-1)
        delta = torch.zeros_like(identity).scatter(0, self.active_flat_indices, self.active_deltas)
        return (identity + delta).reshape_as(self.coefficients)

    def assert_inactive_identity(self):
        expected = torch.zeros_like(self.coefficients)
        expected[..., 1] = 1.
        if not torch.equal(self.coefficients, expected):
            raise RuntimeError("Frozen coefficient-identity buffer changed")
        effective = self.effective_coefficients()
        mask = torch.zeros(8, dtype=torch.bool, device=self.device)
        mask[self.active_flat_indices] = True
        if not torch.equal(effective.reshape(-1)[~mask], expected.reshape(-1)[~mask]):
            raise RuntimeError("An inactive coefficient changed")
        if not torch.isfinite(effective).all():
            raise RuntimeError("Nonfinite effective coefficients")
        return True

    def _injection_hook(self, layer_index, module_index, noise):
        def hook(module, inputs, outputs):
            del module, inputs
            # Last-block-only really means untouched earlier outputs,
            # including no training noise at inactive sites.
            if (layer_index, module_index) not in self.active_sites:
                return outputs
            activation = self._activation(outputs)
            alpha, beta = self.effective_coefficients()[layer_index, module_index].unbind()
            context = self.context_vectors[layer_index, module_index].view(1, 1, -1)
            injected = beta * activation + alpha * context
            if noise and self.noise_scale:
                norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True).detach()
                injected = injected + torch.randn_like(injected) * norm * self.noise_scale
            return (injected,) + outputs[1:] if isinstance(outputs, tuple) else injected
        return hook

    @contextmanager
    def identity_coefficients(self):
        saved = self.active_deltas.detach().clone()
        with torch.no_grad():
            self.active_deltas.zero_()
        try:
            yield
        finally:
            with torch.no_grad():
                self.active_deltas.copy_(saved)

    def adaptation_state(self):
        # The original inference wrapper can load this expanded inference state.
        state = super().adaptation_state()
        state["coefficients"] = self.effective_coefficients().detach().cpu().clone()
        return state

    def variant_training_state(self):
        return {
            "version": VERSION, "sites": self.variant_sites, "learn_beta": self.learn_beta,
            "active_coordinates": [list(row) for row in self.active_coordinates],
            "active_deltas": self.active_deltas.detach().cpu().clone(),
            "active_parameter_count": self.active_deltas.numel(),
            "effective_coefficients": self.effective_coefficients().detach().cpu().clone(),
            "initialization": {"alpha": 0., "beta": 1., "origin": "our identity-init ablation"},
            "author_initialization": {"alpha": .1, "beta": 1.},
            "noise_sites": [list(site) for site in sorted(self.active_sites)],
        }

    def load_adaptation_state(self, state):
        # Explicitly convert an original-style exported coefficient tensor back
        # into the subset of active deltas, validating all fixed coordinates.
        expected = self.coefficients.detach().clone()
        values = state["coefficients"].to(self.device, self.dtype)
        if values.shape != expected.shape:
            raise ValueError("Unexpected coefficient shape")
        mask = torch.zeros(8, dtype=torch.bool, device=self.device)
        mask[self.active_flat_indices] = True
        if not torch.equal(values.reshape(-1)[~mask], expected.reshape(-1)[~mask]):
            raise ValueError("Exported state changes fixed coefficients")
        compatible = dict(state)
        compatible["coefficients"] = expected
        super().load_adaptation_state(compatible)
        with torch.no_grad():
            delta = (values - expected).reshape(-1)
            self.active_deltas.copy_(delta[self.active_flat_indices])
        self.assert_inactive_identity()


def reset_seed(seed, device=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def require_no_hooks(adapter):
    if adapter._hooks_active:
        raise RuntimeError("Wrapper's temporary hooks are still active")
    for _, _, module in adapter._sites():
        if module._forward_hooks:
            raise RuntimeError("An alignment module has unexpected persistent hooks")


def calibrate_variant(adapter, tokens, labels, config, args, destination=None):
    """Same fixed support, weighted BCE and full coefficient identity penalty."""
    check_deadline(args.deadline)
    reset_seed(args.seed, tokens.device)
    trainable = [(name, parameter) for name, parameter in adapter.named_parameters()
                 if parameter.requires_grad]
    if len(trainable) != 1 or trainable[0][1] is not adapter.active_deltas:
        raise RuntimeError("Unexpected trainable parameter outside active_deltas")
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("Base RadZero is not fully frozen")
    targets = torch.tensor(labels, dtype=torch.float32, device=tokens.device)
    criterion, weighting = original_training.make_support_criterion(targets)
    initial_coefficients = adapter.effective_coefficients().detach().cpu().tolist()
    initial_bce = original_training.evaluate_support_loss(
        adapter, tokens, targets, args.grad_batch_size, criterion)
    initial_penalty = float(original_training.coefficient_identity_penalty(
        adapter.effective_coefficients()).detach())
    context_hash = original_training.tensor_sha256(adapter.context_vectors)
    optimizer = torch.optim.AdamW([adapter.active_deltas], lr=config["learning_rate"],
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(tokens) / args.grad_batch_size)
    description = schedule_description(total_steps, args.epochs, len(tokens), args.grad_batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: author_lr_factor(step, total_steps, description["warmup_steps"]))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    gradients_seen = torch.zeros_like(adapter.active_deltas, dtype=torch.bool)
    first_gradient = None
    history, lr_history = [], []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        check_deadline(args.deadline)
        adapter.train()
        order = torch.randperm(len(tokens), generator=generator)
        summed = {"bce": 0., "penalty": 0., "regularization": 0., "total": 0.}
        consumed = 0
        for first in range(0, len(tokens), args.grad_batch_size):
            check_deadline(args.deadline)
            indices = order[first:first + args.grad_batch_size].to(tokens.device)
            require_no_hooks(adapter)
            optimizer.zero_grad(set_to_none=True)
            output = adapter.forward_tokens(tokens.index_select(0, indices), training_noise=True)
            bce = criterion(output["logits"], targets.index_select(0, indices))
            penalty = original_training.coefficient_identity_penalty(adapter.effective_coefficients())
            regularization = config["lambda_reg"] * penalty
            loss = bce + regularization
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite variant training objective")
            loss.backward()
            gradient = adapter.active_deltas.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("Invalid active coefficient gradients")
            if first_gradient is None:
                first_gradient = gradient.detach().cpu().tolist()
            gradients_seen |= gradient != 0
            if any(parameter.grad is not None for parameter in adapter.model.parameters()):
                raise RuntimeError("Frozen model received gradients")
            lr_history.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
            adapter.assert_inactive_identity()
            require_no_hooks(adapter)
            for name, value in (("bce", bce), ("penalty", penalty),
                                ("regularization", regularization), ("total", loss)):
                summed[name] += float(value.detach()) * len(indices)
            consumed += len(indices)
        if consumed != len(tokens):
            raise RuntimeError("Did not consume each support image once")
        entry = {
            "epoch": epoch, "examples_seen": consumed,
            "optimizer_steps": len(lr_history),
            **{name: value / consumed for name, value in summed.items()},
            "effective_coefficients": adapter.effective_coefficients().detach().cpu().tolist(),
            "active_deltas": adapter.active_deltas.detach().cpu().tolist(),
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(entry)
        if destination is not None:
            write_json(destination / "train_history.json", history)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(f"{args.dataset} {config['name']}: epoch {epoch}/{args.epochs}, "
                      f"bce={entry['bce']:.6f}, reg={entry['regularization']:.6f}", flush=True)
        if original_training.tensor_sha256(adapter.context_vectors) != context_hash:
            raise RuntimeError("Fixed context changed during calibration")
    if not bool(gradients_seen.all()):
        raise RuntimeError("Some active scalars never received a nonzero gradient")
    if len(lr_history) != total_steps:
        raise RuntimeError("Optimizer count differs from schedule")
    expected_lrs = [config["learning_rate"] * author_lr_factor(
        step, total_steps, description["warmup_steps"]) for step in range(total_steps)]
    np.testing.assert_allclose(lr_history, expected_lrs, atol=1e-14, rtol=1e-12)
    final_bce = original_training.evaluate_support_loss(
        adapter, tokens, targets, args.grad_batch_size, criterion)
    final_penalty = float(original_training.coefficient_identity_penalty(
        adapter.effective_coefficients()).detach())
    result = {
        "loss_weighting": weighting, "regularization": original_training.regularization_metadata(config["lambda_reg"]),
        "initial_coefficients": initial_coefficients,
        "final_coefficients": adapter.effective_coefficients().detach().cpu().tolist(),
        "initial_support_bce_without_noise": initial_bce,
        "final_support_bce_without_noise": final_bce,
        "initial_coefficient_penalty": initial_penalty, "final_coefficient_penalty": final_penalty,
        "active_parameter_count": adapter.active_deltas.numel(),
        "active_coordinates": [list(row) for row in adapter.active_coordinates],
        "all_active_parameters_received_nonzero_finite_gradients": bool(gradients_seen.all()),
        "first_gradient": first_gradient, "base_parameters_received_gradients": False,
        "inactive_coefficients_exact_identity": adapter.assert_inactive_identity(),
        "context_unchanged": True, "context_sha256": context_hash,
        "schedule": description, "learning_rates_used": lr_history,
        "optimizer_steps": total_steps, "epochs": args.epochs,
        "checkpoint_selection": "final fixed epoch; whole A-F grid fixed before test labels",
        "elapsed_seconds": time.monotonic() - started,
    }
    if destination is not None:
        write_json(destination / "training.json", result)
    return result


def load_verified_support_tokens(directory, support, feature_cache, device,
                                 expected_model_digest=None):
    directory = Path(directory).resolve()
    if not (directory / "support_tokens.npy").is_file():
        directory = directory / feature_cache.dataset
    path = directory / "support_tokens.npy"
    metadata_path = directory / "support_cache.json"
    config_path = directory / "run_config.json"
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    frozen_path = directory / "frozen_model_check.json"
    frozen_check = json.loads(frozen_path.read_text())
    if not frozen_check.get("passed") or frozen_check.get("before_sha256") != frozen_check.get("after_sha256"):
        raise ValueError("Prior support-token run did not verify a frozen base model")
    if expected_model_digest is not None and frozen_check["before_sha256"] != expected_model_digest:
        raise ValueError("Current model tensor digest differs from the support-token model")
    identity = original_training.fingerprint(original_training.spec_identity(support))
    for key in ("support_identity", "context_support_identity", "calibration_support_identity"):
        if config[key] != identity or metadata[key] != identity:
            raise ValueError(f"Support token {key} mismatch")
    if metadata["token_stage"] != "frozen DINO output BEFORE the alignment transformer":
        raise ValueError("Support cache is from the wrong model stage")
    if config["checkpoint"] != feature_cache.manifest["checkpoint"] or config["checkpoint"] != checkpoint_identity():
        raise ValueError("Support token model identity mismatch")
    if config["class_names"] != support["class_names"] or config["prompts"] != support["prompts"]:
        raise ValueError("Support classes/prompts mismatch")
    for source, expected in config["source_sha256"].items():
        source = Path(source)
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Support-token generating source changed: {source}")
    for filename, key in (("config.json", "model_config_sha256"),
                          ("preprocessor_config.json", "preprocessor_sha256")):
        if sha256(MODEL_DIR / filename) != config[key]:
            raise ValueError(f"Support-token model/processor configuration changed: {filename}")
    if Path(metadata["token_path"]).resolve() != path.resolve():
        raise ValueError("Support cache path differs from recorded token path")
    if sha256(path) != metadata["token_sha256"]:
        raise ValueError("Support token content checksum mismatch")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != metadata["shape"] or values.shape != (len(support["image_paths"]), 1370, 768):
        raise ValueError("Unexpected support token dimensions")
    if values.dtype != np.float32 or metadata["dtype"] != "float32" or not np.isfinite(values).all():
        raise ValueError("Support tokens must be finite float32")
    support_csv = feature_cache.split_dir / feature_cache.dataset / "support.csv"
    old_split_digests = [value for key, value in config["split_sha256"].items()
                        if Path(key).name == "support.csv"]
    if old_split_digests != [sha256(support_csv)]:
        raise ValueError("Support token split identity changed")
    provenance = {
        "path": str(path), "token_sha256": metadata["token_sha256"],
        "metadata_sha256": sha256(metadata_path), "source_config_sha256": sha256(config_path),
        "support_identity": identity, "source_sha256_verified": config["source_sha256"],
        "model_checkpoint_verified": config["checkpoint"],
        "frozen_model_check_sha256": sha256(frozen_path),
        "model_tensor_digest": frozen_check["before_sha256"],
        "shape": list(values.shape), "dtype": str(values.dtype), "reused": True,
    }
    return torch.from_numpy(values).to(device), provenance


def verify_export_roundtrip(adapter, tokenizer, tokens, path, metadata=None):
    """Serialize then load into a new ORIGINAL inference wrapper, not subclass."""
    require_no_hooks(adapter)
    adapter.eval()
    with torch.no_grad():
        expected = adapter.forward_tokens(tokens[:2], return_maps=True)
    original_training.atomic_save(path, {
        "format": VERSION, "adaptation_state": adapter.adaptation_state(),
        "variant_training_state": adapter.variant_training_state(),
        **(metadata or {}),
    })
    restored = torch.load(path, map_location="cpu", weights_only=False)
    fresh = RadZeroI2CL(adapter.model, tokenizer, adapter.prompts)
    fresh.load_adaptation_state(restored["adaptation_state"])
    fresh.eval()
    with torch.no_grad():
        actual = fresh.forward_tokens(tokens[:2], return_maps=True)
    errors = {}
    for key in ("logits", "similarity_scores"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-6)
        errors[key] = float((actual[key] - expected[key]).abs().max())
    require_no_hooks(adapter)
    require_no_hooks(fresh)
    return {"passed": True, "reloaded_class": "original RadZeroI2CL",
            "new_wrapper_instance": True, "max_absolute_errors": errors}


def strict_full_baseline_check(actual, expected, atol=1e-6):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Invalid full-test baseline array")
    maximum = float(np.abs(actual.astype(np.float64) - expected).max(initial=0))
    if maximum > atol:
        raise ValueError(f"Frozen baseline mismatch {maximum:.9g} > {atol}")
    return maximum


def score_variants_from_pixels(base, adapters, pixels, deadline=None):
    """One and only one image-encoder call for this test batch."""
    with torch.no_grad():
        check_deadline(deadline)
        tokens = base.encode_images(pixels)
        require_no_hooks(base)
        baseline = base.forward_tokens(tokens, inject=False)["logits"]
        logits = {}
        for name, adapter in adapters.items():
            check_deadline(deadline)
            require_no_hooks(adapter)
            logits[name] = adapter.forward_tokens(tokens)["logits"]
            require_no_hooks(adapter)
    return baseline, logits


def snapshot_sources(output, cache):
    modules = [
        Path(__file__).resolve(), Path(original_training.__file__).resolve(),
        Path(__file__).with_name("i2cl_model.py"),
        Path(__file__).with_name("overnight_linear_residual.py"),
        Path(__file__).with_name("overnight_cache_methods.py"),
        Path(__file__).with_name("i2cl_data.py"),
        Path(__file__).with_name("infer_batch.py"),
        Path(__file__).with_name("infer.py"),
        Path(__file__).with_name("batch_data.py"),
        Path(__file__).with_name("batch_metrics.py"),
        AUTHOR_WRAPPER, AUTHOR_CONFIG,
    ]
    root = output / "source_snapshot"
    root.mkdir()
    records = {}
    for index, path in enumerate(modules):
        path = path.resolve()
        destination = root / f"{index:02d}_{path.name}"
        destination.write_bytes(path.read_bytes())
        records[str(path)] = sha256(path)
    for path in sorted(MODEL_DIR.glob("*.py")):
        records[str(path)] = sha256(path)
    # The frozen cache provides additional model-source provenance.
    for path, expected in cache.manifest["source_sha256"].items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen feature generating source changed: {path}")
    return records


def run(args):
    check_deadline(args.deadline)
    cache = FrozenCache(args.feature_cache)
    if cache.dataset != args.dataset:
        raise ValueError("Dataset differs from frozen feature cache")
    if args.eval_batch_size != cache.manifest["batch_size"]:
        raise ValueError("Use the frozen-cache batch size for strict identical-baseline comparison")
    support = cache.support()
    test = cache.metadata["test"]  # no test labels are present here
    if cache.manifest["checkpoint"] != checkpoint_identity():
        raise ValueError("Current model differs from frozen-cache model")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_sources(output, cache)
    configs = []
    for name in args.variants:
        config = {"name": name, **VARIANTS[name],
                  "initial_alpha": 0., "initial_beta": 1.,
                  "initialization_origin": "our identity-init ablation; author uses alpha=.1",
                  "epochs": args.epochs, "grad_batch_size": args.grad_batch_size,
                  "noise_scale": args.noise_scale, "weight_decay": args.weight_decay,
                  "seed": args.seed, "schedule": "author_formula_5pct_warmup_cosine"}
        config["effective_trainable_parameter_count"] = (
            (4 if config["sites"] == "all" else 2) * (2 if config["learn_beta"] else 1))
        configs.append(config)
    run_config = {
        "version": VERSION, "dataset": args.dataset, "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "variants": configs, "source_sha256": source_hashes,
        "checkpoint": cache.manifest["checkpoint"], "cache_manifest_sha256": sha256(cache.path),
        "support_sha256": cache.metadata["support"]["csv_sha256"],
        "test_sha256": cache.metadata["test"]["csv_sha256"],
        "class_names": support["class_names"], "prompts": support["prompts"],
        "support_count": len(support["image_paths"]), "test_count": len(test["image_paths"]),
        "support_identity": original_training.fingerprint(original_training.spec_identity(support)),
        "split_metadata": {key: support["metadata"][key]
                           for key in ("split_fingerprint", "root_split_fingerprint")},
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "preprocessor_sha256": sha256(MODEL_DIR / "preprocessor_config.json"),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": "float32; TF32 disabled",
        "test_encoder_policy": "one DINO pass per test batch; baseline and variants reuse those tokens",
        "selection": "predeclared A-F grid; no epoch/hyperparameter choice using test labels",
        "exploratory": True,
        "test_labels_read_only_after_all_training_and_predictions": True,
    }
    write_json(output / "run_config.json", run_config)
    write_json(output / "status.json", {"status": "loading_model"})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    reset_seed(args.seed, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model, processor, tokenizer = load_model(device)
    model = model.float().requires_grad_(False).eval()
    base = RadZeroI2CL(model, tokenizer, support["prompts"], noise_scale=args.noise_scale)
    base.coefficients.requires_grad_(False)
    before_hash = original_training.tensor_digest(model)
    try:
        tokens, cache_audit = load_verified_support_tokens(
            args.support_cache, support, cache, device, expected_model_digest=before_hash)
        check_deadline(args.deadline)
        # Independent live encoder check on a small support batch.
        support_loader = original_training.image_loader(support, processor, args, 2)
        first_pixels, _, indices = next(iter(support_loader))
        del support_loader
        if indices.tolist() != [0, 1]:
            raise RuntimeError("Support verification row order changed")
        with torch.no_grad():
            fresh_tokens = base.encode_images(first_pixels.to(device))
        torch.testing.assert_close(fresh_tokens, tokens[:2], atol=2e-5, rtol=1e-4)
        cache_audit["fresh_token_check"] = {
            "count": 2, "max_abs_error": float((fresh_tokens - tokens[:2]).abs().max()),
            "atol": 2e-5, "rtol": 1e-4,
        }
        write_json(output / "support_cache_audit.json", cache_audit)
        base.build_context(tokens, batch_size=args.grad_batch_size)
        context_hash = original_training.tensor_sha256(base.context_vectors)
        adapters, training = {}, {}
        for config in configs:
            check_deadline(args.deadline)
            name = config["name"]
            destination = output / name
            destination.mkdir()
            adapter = ConservativeI2CL(
                model, tokenizer, support["prompts"], sites=config["sites"],
                learn_beta=config["learn_beta"], noise_scale=args.noise_scale)
            with torch.no_grad():
                adapter.context_vectors.copy_(base.context_vectors)
            adapter.context_ready = True
            adapter.context_sample_count = len(tokens)
            if adapter.active_deltas.numel() != config["effective_trainable_parameter_count"]:
                raise RuntimeError("Configured and actual parameter counts differ")
            adapter.eval()
            with torch.no_grad():
                reference = base.forward_tokens(tokens[:2], inject=False, return_maps=True)
                identity = adapter.forward_tokens(tokens[:2], return_maps=True)
            identity_errors = {}
            for key in reference:
                torch.testing.assert_close(identity[key], reference[key], atol=0., rtol=0.)
                identity_errors[key] = float((identity[key] - reference[key]).abs().max())
            write_json(destination / "identity_check.json", {
                "passed": True, "max_errors": identity_errors, "alpha": 0., "beta": 1.})
            write_json(output / "status.json", {"status": "training", "variant": name,
                                               "completed_variants": list(adapters)})
            fitted = calibrate_variant(adapter, tokens, support["labels"], config, args, destination)
            if original_training.tensor_digest(model) != before_hash:
                raise RuntimeError("Frozen model changed during variant training")
            if original_training.tensor_sha256(adapter.context_vectors) != context_hash:
                raise RuntimeError("Variant context differs from the shared support-only context")
            reload_check = verify_export_roundtrip(
                adapter, tokenizer, tokens, destination / "i2cl_variant.pt",
                metadata={
                    "dataset": args.dataset, "split_dir": str(cache.split_dir),
                    "model_identity": cache.manifest["checkpoint"],
                    "support_sha256": cache.metadata["support"]["csv_sha256"],
                    "class_names": support["class_names"], "prompts": support["prompts"],
                    "run_config": {**run_config, "variant": config},
                    "training": fitted,
                    "support_identity": run_config["support_identity"],
                    "context_support_identity": run_config["support_identity"],
                    "calibration_support_identity": run_config["support_identity"],
                })
            write_json(destination / "checkpoint_reload_check.json", reload_check)
            write_json(destination / "frozen_check.json", {
                "passed": True, "before_sha256": before_hash, "after_sha256": before_hash,
                "active_parameter_count": adapter.active_deltas.numel(),
                "base_trainable_count": 0,
            })
            training[name] = fitted
            adapter.eval()
            adapters[name] = adapter
        del tokens, first_pixels, fresh_tokens
        if device.type == "cuda":
            torch.cuda.empty_cache()
        check_deadline(args.deadline)
        baseline_expected = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float32)
        predictions = {name: np.empty_like(baseline_expected) for name in adapters}
        computed_baseline = np.empty_like(baseline_expected)
        loader = original_training.image_loader(test, processor, args, args.eval_batch_size)
        offset = 0
        baseline_max_error = 0.
        started_eval = time.monotonic()
        for pixels, _, indices in loader:
            check_deadline(args.deadline)
            stop = offset + len(pixels)
            if indices.tolist() != list(range(offset, stop)):
                raise RuntimeError("Test image order changed")
            baseline, outputs = score_variants_from_pixels(
                base, adapters, pixels.to(device), deadline=args.deadline)
            actual = baseline.cpu().numpy()
            error = strict_full_baseline_check(actual, baseline_expected[offset:stop])
            baseline_max_error = max(baseline_max_error, error)
            computed_baseline[offset:stop] = actual
            for name, values in outputs.items():
                predictions[name][offset:stop] = values.cpu().numpy()
            offset = stop
            write_json(output / "status.json", {
                "status": "evaluating_shared_encoder", "completed_images": offset,
                "total_images": len(baseline_expected),
                "images_per_second": offset / max(.001, time.monotonic() - started_eval),
                "baseline_max_abs_error": baseline_max_error,
            })
            if offset == len(pixels) or offset % (args.eval_batch_size * 32) == 0 or offset == len(baseline_expected):
                print(f"{args.dataset}: 6-variant evaluation {offset}/{len(baseline_expected)}", flush=True)
        del loader
        if offset != len(baseline_expected):
            raise RuntimeError("Test inference incomplete")
        for name, values in predictions.items():
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite variant predictions")
            np.save(output / name / "predictions.npy", values, allow_pickle=False)
        np.save(output / "baseline_predictions.npy", computed_baseline, allow_pickle=False)
        write_json(output / "baseline_check.json", {
            "passed": True, "all_test_images_checked": offset,
            "max_absolute_error": baseline_max_error, "atol": 1e-6, "rtol": 0.,
            "bitwise_equal": bool(np.array_equal(computed_baseline, baseline_expected)),
        })
        if original_training.tensor_digest(model) != before_hash:
            raise RuntimeError("Base model changed during evaluation")
        # This is the first point at which test labels are loaded.
        trials = []
        for config in configs:
            check_deadline(args.deadline)
            name = config["name"]
            destination = output / name
            result = evaluate_test_only(
                cache.split_dir, args.dataset, predictions[name], computed_baseline,
                cache.metadata["test"], destination)
            trial = {
                "name": name, "config": config, "metrics": result["metrics"],
                "baseline_metrics": result["baseline_metrics"],
                "delta_pp": {key: None if value is None else 100 * value
                             for key, value in result["delta"].items()},
                "training": training[name],
                "prediction_sha256": sha256(destination / "predictions.npy"),
                "checkpoint_sha256": sha256(destination / "i2cl_variant.pt"),
                "base_unchanged": True, "context_sha256": context_hash,
            }
            write_json(destination / "result.json", trial)
            trials.append(trial)
            write_json(output / "summary.json", {
                "dataset": args.dataset, "version": VERSION,
                "exploratory": True, "baseline_metrics": result["baseline_metrics"],
                "trials": trials, "all_test_images": offset,
                "test_encoder_passes": 1, "source_sha256": source_hashes,
            })
        write_json(output / "status.json", {
            "status": "complete", "completed_variants": [item["name"] for item in trials],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    except DeadlineReached as error:
        write_json(output / "status.json", {"status": "deadline", "reason": str(error)})
        raise
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "reason": repr(error)})
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA"))
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--support-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--deadline", type=parse_deadline, required=True)
    args = parser.parse_args()
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants")
    if min(args.epochs, args.grad_batch_size, args.eval_batch_size) < 1 or args.workers < 0:
        parser.error("Epochs/batches must be positive; workers nonnegative")
    if not 1 <= args.threads <= 4:
        parser.error("threads must be between 1 and 4")
    if any(not math.isfinite(value) or value < 0 for value in (args.noise_scale, args.weight_decay)):
        parser.error("Noise/weight decay must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        try:
            run(args)
        except DeadlineReached as error:
            print(str(error), flush=True)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
