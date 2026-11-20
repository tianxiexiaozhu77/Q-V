#!/usr/bin/env python3
"""Diagnose I2CL history/ablations, optionally with matched scheduler retraining.

The label-covered test subset is deliberately diagnostic. Comparing historical
epochs on this subset does not provide a new unbiased model-selection result.
All states use the original fixed context and identical cached DINO tokens.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from infer_batch import ASSETS_DIR, MODEL_DIR, checkpoint_identity, load_model, now, write_json
from i2cl_data import load_spec
from i2cl_model import RadZeroI2CL
from batch_data import DATASETS
from batch_metrics import compute_metrics
from train_i2cl import (
    assert_split_contract, fingerprint, image_loader, sha256, spec_identity,
    tensor_digest, tensor_sha256,
)

import numpy as np
import torch


DIAGNOSTIC_WARNING = (
    "Post-hoc diagnosis only. Test labels are used for coverage-aware sampling and "
    "historical epochs are compared on the same test subset. These results must not "
    "be reported as unbiased validation-selected or new best test performance."
)
HISTORY_EPOCHS = (1, 5, 10, 20, 50, 100)


def subset_spec(spec, indices):
    selected = dict(spec)
    for key in ("image_paths", "ids", "groups"):
        selected[key] = [spec[key][index] for index in indices]
    selected["labels"] = np.asarray(spec["labels"])[indices]
    return selected


def choose_test_subset(spec, limit, seed):
    """Cover both states of every class, then fill uniformly without replacement."""
    labels = np.asarray(spec["labels"], dtype=np.int8)
    count = min(limit, len(labels))
    if count == len(labels):
        indices = np.arange(len(labels), dtype=np.int64)
        quotas = np.asarray([
            [min(5, int((labels[:, col] == state).sum())) for state in (0, 1)]
            for col in range(labels.shape[1])
        ])
    else:
        rng = np.random.default_rng(seed)
        selected = set()
        quotas = np.asarray([
            [min(5, int((labels[:, col] == state).sum())) for state in (0, 1)]
            for col in range(labels.shape[1])
        ])
        # Rare target states are covered first; shuffle order within each state.
        targets = sorted(
            ((int((labels[:, col] == state).sum()), col, state)
             for col in range(labels.shape[1]) for state in (0, 1)),
        )
        for _, col, state in targets:
            covered = sum(labels[index, col] == state for index in selected)
            needed = max(0, int(quotas[col, state]) - covered)
            if needed:
                available = [
                    int(index) for index in np.flatnonzero(labels[:, col] == state)
                    if int(index) not in selected
                ]
                rng.shuffle(available)
                selected.update(available[:needed])
        if len(selected) > count:
            raise ValueError(
                f"--limit-test={limit} is too small for the fixed per-state coverage "
                f"protocol ({len(selected)} images required); increase it"
            )
        available = np.asarray([
            index for index in range(len(labels)) if index not in selected
        ], dtype=np.int64)
        fill = rng.choice(available, size=count - len(selected), replace=False)
        selected.update(int(index) for index in fill)
        indices = np.asarray(sorted(selected), dtype=np.int64)
    selected = subset_spec(spec, indices)
    for col in range(labels.shape[1]):
        for state in (0, 1):
            if int((selected["labels"][:, col] == state).sum()) < quotas[col, state]:
                raise RuntimeError("Diagnostic subset did not satisfy its coverage quota")
    return selected, indices, {
        "algorithm": (
            "Rare class/state first; include enough seeded random examples to obtain "
            "min(5, available) images for each class state, then fill uniformly "
            "without replacement; return original held-out row order"
        ),
        "seed": seed,
        "requested_limit": limit,
        "full_holdout_count": len(labels),
        "selected_count": len(indices),
        "per_class_negative_positive_minimum": quotas.tolist(),
        "per_class_selected_positive_counts": selected["labels"].sum(0).tolist(),
        "per_class_full_positive_counts": labels.sum(0).tolist(),
        "bias": (
            "Coverage sampling overrepresents rare findings. Absolute subset "
            "metrics/PR AUC are not estimates of the full-test prevalence."
        ),
    }


def coefficient_states(checkpoint, history):
    by_epoch = {entry["epoch"]: entry for entry in history}
    if len(by_epoch) != len(history):
        raise ValueError("Duplicate epochs in training history")
    if any(epoch not in by_epoch for epoch in HISTORY_EPOCHS):
        raise ValueError("This diagnostic expects the completed 100-epoch experiment")
    final = checkpoint["adaptation_state"]["coefficients"].detach().cpu().float()
    expected = torch.tensor(by_epoch[100]["coefficients"], dtype=torch.float32)
    torch.testing.assert_close(final, expected, atol=0, rtol=0)
    baseline = torch.zeros_like(final)
    baseline[..., 1] = 1
    initial = baseline.clone()
    initial[..., 0] = 0.1
    states = {"baseline": baseline, "initial": initial}
    for epoch in HISTORY_EPOCHS:
        states[f"epoch_{epoch:03d}"] = torch.tensor(
            by_epoch[epoch]["coefficients"], dtype=torch.float32
        )
    alpha_only = final.clone()
    alpha_only[..., 1] = 1
    beta_only = final.clone()
    beta_only[..., 0] = 0
    states["final_alpha_only"] = alpha_only
    states["final_beta_only"] = beta_only
    for value in states.values():
        if value.shape != (2, 2, 2) or not torch.isfinite(value).all():
            raise ValueError("Invalid recorded diagnostic coefficients")
    return states


def coefficient_summary(coefficients):
    values = np.asarray(coefficients, dtype=np.float32)
    alpha, beta = values[..., 0], values[..., 1]
    identity = np.zeros_like(values)
    identity[..., 1] = 1
    return {
        "coefficients": values.tolist(),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "negative_alpha_count": int((alpha < 0).sum()),
        "negative_beta_count": int((beta < 0).sum()),
        "distance_from_identity_l2": float(np.linalg.norm(values - identity)),
    }


def encode_test(wrapper, spec, processor, args, device):
    chunks = []
    offset = 0
    started = time.monotonic()
    loader = image_loader(spec, processor, args, args.batch_size)
    with torch.no_grad():
        for pixels, _, indices in loader:
            if not np.array_equal(indices.numpy(), np.arange(offset, offset + len(pixels))):
                raise RuntimeError("Diagnostic test order changed during encoding")
            tokens = wrapper.encode_images(pixels.to(device, non_blocking=True))
            chunks.append(tokens.cpu())
            offset += len(tokens)
            if offset == len(tokens) or offset % 128 == 0 or offset == len(spec["image_paths"]):
                print(f"{args.dataset}: cached diagnostic test {offset}/{len(spec['image_paths'])}",
                      flush=True)
    del loader
    values = torch.cat(chunks)
    if tuple(values.shape) != (len(spec["image_paths"]), 1370, 768):
        raise RuntimeError("Unexpected diagnostic DINO token shape")
    return values, {"shape": list(values.shape), "dtype": "float32",
                    "elapsed_seconds": time.monotonic() - started,
                    "cache": "RAM only; original frozen DINO output before alignment"}


def score_tokens(wrapper, tokens, batch_size, state_name):
    arrays = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start:start + batch_size].to(wrapper.device)
            logits = wrapper.forward_tokens(
                batch, inject=state_name != "baseline", training_noise=False
            )["logits"]
            arrays.append(logits.cpu().numpy())
    return np.concatenate(arrays)


def summarize_scores(spec, logits):
    metrics = compute_metrics(spec, logits)
    metrics["bce"] = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits).double(),
        torch.tensor(np.asarray(spec["labels"]), dtype=torch.float64),
    ))
    metrics["mean_logit_by_class"] = logits.mean(0).astype(float).tolist()
    metrics["mean_sigmoid_by_class"] = torch.sigmoid(
        torch.from_numpy(logits).double()
    ).mean(0).tolist()
    metrics["positive_fraction_by_class"] = np.asarray(spec["labels"]).mean(0).tolist()
    return metrics


def compare_saved_predictions(path, spec, logits):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {row["image_path"]: row for row in rows}
    expected = []
    for image_path, labels in zip(spec["image_paths"], spec["labels"]):
        row = lookup[image_path]
        if [int(row[f"label:{name}"]) for name in spec["class_names"]] != labels.tolist():
            raise RuntimeError("Diagnostic targets differ from saved experiment")
        expected.append([float(row[f"logit:{name}"]) for name in spec["class_names"]])
    expected = np.asarray(expected)
    np.testing.assert_allclose(logits, expected, atol=2e-4, rtol=1e-4)
    return {"passed": True, "images": len(expected),
            "max_absolute_error": float(np.max(np.abs(logits - expected)))}


def compare_schedulers(wrapper, support_tokens, test_tokens, support, test, config,
                       original_state, original_scores, args):
    """Retrain matched diagnostic branches; only the LR schedule differs."""
    training_args = config["arguments"]
    epochs = 100
    batch_size = int(training_args["grad_batch_size"])
    base_lr = float(training_args["learning_rate"])
    weight_decay = float(training_args["weight_decay"])
    noise_scale = float(training_args["noise_scale"])
    training_seed = 42
    steps_per_epoch = math.ceil(len(support_tokens) / batch_size)
    total_steps = epochs * steps_per_epoch
    # The current supports have even counts and batch size 2; ceil equals the
    # upstream floor expression. Reject ambiguity rather than change its formula.
    if len(support_tokens) % batch_size:
        raise ValueError("Exact upstream scheduler comparison requires divisible support batches")
    warmup_steps = int((0.05 * epochs) * (len(support_tokens) // batch_size))
    if warmup_steps < 1:
        raise ValueError("The upstream warmup formula requires at least one warmup step")
    # Keep the cached pool in CPU RAM. Only the current minibatch uses VRAM,
    # including when running beside existing GPU jobs with an allocator cap.
    tokens = support_tokens.cpu()
    labels = torch.tensor(np.asarray(support["labels"]), dtype=torch.float32)
    context_hash = tensor_sha256(wrapper.context_vectors)
    reports = {}
    prediction_arrays = {}
    for branch in ("fixed_lr", "official_warmup_cosine"):
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        if wrapper.device.type == "cuda":
            torch.cuda.manual_seed_all(training_seed)
        generator = torch.Generator(device="cpu").manual_seed(training_seed)
        with torch.no_grad():
            wrapper.coefficients[..., 0].fill_(0.1)
            wrapper.coefficients[..., 1].fill_(1)
        wrapper.coefficients.grad = None
        wrapper.noise_scale = noise_scale
        optimizer = torch.optim.AdamW(
            [wrapper.coefficients], lr=base_lr, weight_decay=weight_decay
        )
        scheduler = None
        if branch == "official_warmup_cosine":
            def lr_multiplier(step):
                return (
                    min(1.0, step / warmup_steps)
                    * (1 + math.cos(math.pi * step / total_steps)) / 2
                    if step > warmup_steps else step / warmup_steps
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        rows = []
        gradient_seen = torch.zeros_like(wrapper.coefficients, dtype=torch.bool)
        branch_started = time.monotonic()
        for epoch in range(1, epochs + 1):
            wrapper.train()
            order = torch.randperm(len(tokens), generator=generator)
            loss_sum = 0.0
            rates = []
            for start in range(0, len(tokens), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = wrapper.forward_tokens(
                    tokens.index_select(0, indices).to(wrapper.device), training_noise=True
                )["logits"]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.index_select(0, indices).to(wrapper.device)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite scheduler diagnostic loss")
                loss.backward()
                gradient = wrapper.coefficients.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError("Invalid scheduler diagnostic gradients")
                gradient_seen |= gradient != 0
                rates.append(float(optimizer.param_groups[0]["lr"]))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if not torch.isfinite(wrapper.coefficients).all():
                    raise RuntimeError("Invalid scheduler diagnostic coefficients")
                loss_sum += float(loss.detach()) * len(indices)
            rows.append({
                "epoch": epoch, "examples_seen": len(tokens),
                "optimizer_steps": steps_per_epoch,
                "train_bce_with_noise": loss_sum / len(tokens),
                "lr_first_used": rates[0], "lr_last_used": rates[-1],
                "lr_next": float(optimizer.param_groups[0]["lr"]),
                **coefficient_summary(wrapper.coefficients.detach().cpu().numpy()),
            })
            if epoch == 1 or epoch % 10 == 0:
                write_json(args.output / f"controlled_{branch}_history.json", rows)
                print(f"{args.dataset}: {branch} epoch {epoch}/100, "
                      f"BCE={rows[-1]['train_bce_with_noise']:.6f}, "
                      f"lr_next={rows[-1]['lr_next']:.8f}", flush=True)
        if not gradient_seen.all() or any(p.grad is not None for p in wrapper.model.parameters()):
            raise RuntimeError("Scheduler comparison failed frozen/gradient checks")
        if tensor_sha256(wrapper.context_vectors) != context_hash:
            raise RuntimeError("Scheduler diagnostic altered its fixed context")
        wrapper.eval()
        support_scores = score_tokens(wrapper, support_tokens, args.batch_size, branch)
        test_scores = score_tokens(wrapper, test_tokens, args.batch_size, branch)
        state = wrapper.adaptation_state()
        report = {
            "branch": branch, "seed": training_seed,
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps, "warmup_steps": warmup_steps if scheduler else 0,
            "base_lr": base_lr, "weight_decay": weight_decay, "noise_scale": noise_scale,
            "batch_size": batch_size, "all_coefficient_gradients_seen": True,
            "context_sha256": context_hash,
            "support": summarize_scores(support, support_scores),
            "test": summarize_scores(test, test_scores),
            **coefficient_summary(state["coefficients"].numpy()),
            "elapsed_seconds": time.monotonic() - branch_started,
            "sampling": "Same seeded randperm of all support images once per epoch in both branches",
            "rng": "Python/NumPy/Torch/CUDA RNGs reset to 42 immediately before each branch",
            "difference_from_original_final_coefficients_l2": float(torch.linalg.vector_norm(
                state["coefficients"] - original_state["coefficients"].cpu()
            )),
            "max_absolute_logit_difference_from_original_final": float(
                np.max(np.abs(test_scores - original_scores))
            ),
        }
        reports[branch] = report
        prediction_arrays[branch + "_support_logits"] = support_scores
        prediction_arrays[branch + "_test_logits"] = test_scores
        write_json(args.output / f"controlled_{branch}_history.json", rows)
        torch.save({
            "purpose": DIAGNOSTIC_WARNING,
            "adaptation_state": state,
            "model_identity": config["checkpoint"],
            "support_identity": config["support_identity"],
            "branch": branch, "report": report,
        }, args.output / f"controlled_{branch}.pt")
        wrapper.coefficients.grad = None
    reports["comparison"] = {
        "only_controlled_difference": "Learning-rate scheduler; all other inputs and RNG reset match",
        "upstream_source": str(Path(__file__).resolve().parents[2] / "I2CL/wrapper.py"),
        "upstream_source_sha256": sha256(
            Path(__file__).resolve().parents[2] / "I2CL/wrapper.py"
        ),
        "upstream_formula": (
            "step / warmup_steps if step <= warmup_steps else "
            "min(1, step/warmup_steps)*(1+cos(pi*step/total_steps))/2"
        ),
        "scheduler_order": "LambdaLR constructed before training; optimizer.step then scheduler.step",
        "test_delta_pp_cosine_minus_fixed": {
            key: 100 * (reports["official_warmup_cosine"]["test"][key]
                        - reports["fixed_lr"]["test"][key])
            for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")
        },
        "original_run_reproduction_caveat": (
            "Both branches start from a fresh RNG reset. Original training RNG also "
            "depended on model loading and data-loader operations, so fixed_lr is "
            "not assumed to exactly reproduce the original run; differences are recorded."
        ),
    }
    np.savez_compressed(args.output / "controlled_scheduler_predictions.npz",
                        **prediction_arrays,
                        support_labels=np.asarray(support["labels"]),
                        test_labels=np.asarray(test["labels"]))
    write_json(args.output / "controlled_scheduler_comparison.json", reports)
    return reports


def run(args):
    started = time.monotonic()
    checkpoint_path = args.checkpoint
    directory = checkpoint_path.parent
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["run_config"]
    history_path = directory / "train_history.json"
    history = json.loads(history_path.read_text())
    result_source = json.loads((directory / "result.json").read_text())
    if checkpoint["dataset"] != args.dataset or result_source["status"] != "complete":
        raise RuntimeError("Checkpoint does not belong to a completed requested dataset")
    if result_source["is_smoke_test"] or result_source["training"]["epochs"] != 100:
        raise RuntimeError("Use the completed full 100-epoch experiment")
    if checkpoint["model_identity"] != checkpoint_identity():
        raise RuntimeError("Original model identity changed")
    for filename, key in (("config.json", "model_config_sha256"),
                          ("preprocessor_config.json", "preprocessor_sha256")):
        if sha256(MODEL_DIR / filename) != config[key]:
            raise RuntimeError("Original model/preprocessor configuration changed")
    for path, expected_hash in config["source_sha256"].items():
        if sha256(Path(path)) != expected_hash:
            raise RuntimeError("Training source changed since the experiment: " + path)
    split_dir = Path(checkpoint["split_dir"])
    support = load_spec(split_dir, args.dataset, split="support")
    full_test = load_spec(split_dir, args.dataset, split="test")
    assert_split_contract(support, full_test)
    if support["metadata"]["support_sha256"] != checkpoint["support_sha256"]:
        raise RuntimeError("Support split changed")
    for key in ("split_fingerprint", "root_split_fingerprint"):
        if full_test["metadata"][key] != config["split_metadata"][key]:
            raise RuntimeError("Test split changed")
    if fingerprint(spec_identity(support)) != checkpoint["support_identity"]:
        raise RuntimeError("Support identity changed")
    states = coefficient_states(checkpoint, history)
    test, test_indices, sampling = choose_test_subset(full_test, args.limit_test, args.seed)
    support_token_path = directory / "support_tokens.npy"
    cache_manifest = json.loads((directory / "support_cache.json").read_text())
    if sha256(support_token_path) != cache_manifest["token_sha256"]:
        raise RuntimeError("Saved support DINO tokens changed")
    if cache_manifest["support_identity"] != checkpoint["support_identity"]:
        raise RuntimeError("Saved support DINO tokens use different samples")
    support_tokens = torch.from_numpy(np.load(support_token_path, allow_pickle=False))
    if tuple(support_tokens.shape) != (len(support["image_paths"]), 1370, 768):
        raise RuntimeError("Saved support token shape differs")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "dataset": args.dataset,
        "purpose": DIAGNOSTIC_WARNING,
        "created_at": now(),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_history_sha256": sha256(history_path),
        "source_run_fingerprint": config["fingerprint"],
        "diagnostic_code_sha256": sha256(Path(__file__)),
        "support_token_sha256": cache_manifest["token_sha256"],
        "support_identity": checkpoint["support_identity"],
        "context_sha256": tensor_sha256(checkpoint["adaptation_state"]["context_vectors"]),
        "state_names": list(states),
        "sampling": sampling,
        "test_indices": test_indices.tolist(),
        "test_spec": spec_identity(test),
        "support_spec": spec_identity(support),
        "noise": "disabled for every diagnostic state",
        "training": (
            "Optional matched diagnostic branches fixed_lr and official_warmup_cosine"
            if args.compare_schedulers else "none; no optimizer and no new saved model"
        ),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_memory_fraction": args.cuda_memory_fraction,
    }
    write_json(args.output / "diagnostic_manifest.json", manifest)
    write_json(args.output / "coefficient_trajectory.json", [
        dict(epoch=row["epoch"], training_bce_with_noise=row["train_bce_with_noise"],
             **coefficient_summary(row["coefficients"]))
        for row in history
    ])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if device.type == "cuda" and args.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    model, processor, tokenizer = load_model(device)
    wrapper = RadZeroI2CL(model, tokenizer, support["prompts"])
    wrapper.load_adaptation_state(checkpoint["adaptation_state"])
    wrapper.eval()
    before_base = tensor_digest(model)
    before_context = tensor_sha256(wrapper.context_vectors)
    test_tokens, token_report = encode_test(wrapper, test, processor, args, device)
    support_logits, test_logits, summaries = [], [], {}
    for state_name, coefficients in states.items():
        with torch.no_grad():
            wrapper.coefficients.copy_(coefficients.to(device))
        support_scores = score_tokens(wrapper, support_tokens, args.batch_size, state_name)
        test_scores = score_tokens(wrapper, test_tokens, args.batch_size, state_name)
        support_logits.append(support_scores)
        test_logits.append(test_scores)
        summaries[state_name] = {
            **coefficient_summary(coefficients.numpy()),
            "support": summarize_scores(support, support_scores),
            "test": summarize_scores(test, test_scores),
        }
        baseline = summaries["baseline"]["test"]
        summaries[state_name]["test_delta_pp_vs_baseline"] = {
            key: 100 * (summaries[state_name]["test"][key] - baseline[key])
            for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")
            if summaries[state_name]["test"][key] is not None and baseline[key] is not None
        }
        write_json(args.output / "progress.json", {
            "status": "diagnosing", "dataset": args.dataset,
            "completed_states": list(summaries), "total_states": len(states),
            "updated_at": now(),
        })
        print(
            f"{args.dataset} {state_name}: support BCE={summaries[state_name]['support']['bce']:.6f}, "
            f"test BCE={summaries[state_name]['test']['bce']:.6f}, "
            f"test AUROC={summaries[state_name]['test']['macro_auroc']:.6f}", flush=True,
        )
    if tensor_digest(model) != before_base or tensor_sha256(wrapper.context_vectors) != before_context:
        raise RuntimeError("Diagnostic changed frozen model or context")
    if any(parameter.grad is not None for parameter in wrapper.parameters()):
        raise RuntimeError("Diagnostic unexpectedly created gradients")
    states_order = list(states)
    final_index = states_order.index("epoch_100")
    checks = {
        "base_unchanged": True,
        "context_unchanged": True,
        "history_and_ablation_scoring_no_gradients": True,
        "baseline_saved_logits": compare_saved_predictions(
            directory / "baseline_predictions.csv", test, test_logits[0]
        ),
        "final_saved_logits": compare_saved_predictions(
            directory / "predictions.csv", test, test_logits[final_index]
        ),
    }
    for state_name, key in (
        ("initial", "initial_support_bce_without_noise"),
        ("epoch_100", "final_support_bce_without_noise"),
    ):
        actual = summaries[state_name]["support"]["bce"]
        expected = result_source["training"][key]
        if abs(actual - expected) > 2e-5:
            raise RuntimeError(f"Support BCE did not reproduce {state_name}: {actual} vs {expected}")
        checks[f"{state_name}_support_bce_reproduced"] = {
            "passed": True, "absolute_error": abs(actual - expected),
        }
    scheduler_comparison = None
    if args.compare_schedulers:
        scheduler_comparison = compare_schedulers(
            wrapper, support_tokens, test_tokens, support, test, config,
            checkpoint["adaptation_state"], test_logits[final_index], args
        )
        if tensor_digest(model) != before_base or tensor_sha256(wrapper.context_vectors) != before_context:
            raise RuntimeError("Scheduler diagnostic changed frozen model or context")
        checks["scheduler_base_and_context_unchanged"] = True
    np.savez_compressed(
        args.output / "predictions.npz",
        state_names=np.asarray(states_order),
        support_logits=np.stack(support_logits), test_logits=np.stack(test_logits),
        support_labels=np.asarray(support["labels"]), test_labels=np.asarray(test["labels"]),
        test_indices=test_indices,
    )
    with (args.output / "diagnostic_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "split", "samples", "bce", "macro_auroc", "macro_auprc",
                         "micro_auroc", "micro_auprc", "distance_from_identity_l2"])
        for state_name, report in summaries.items():
            for split in ("support", "test"):
                metrics = report[split]
                writer.writerow([
                    state_name, split, metrics["sample_count"], metrics["bce"],
                    metrics["macro_auroc"], metrics["macro_auprc"], metrics["micro_auroc"],
                    metrics["micro_auprc"], report["distance_from_identity_l2"],
                ])
    result = {
        "status": "complete", "dataset": args.dataset, "purpose": DIAGNOSTIC_WARNING,
        "support_count": len(support["image_paths"]), "test_count": len(test["image_paths"]),
        "sampling": sampling, "states": summaries, "checks": checks,
        "scheduler_comparison": scheduler_comparison,
        "test_token_cache": token_report,
        "elapsed_seconds": time.monotonic() - started, "completed_at": now(),
    }
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {
        "status": "complete", "dataset": args.dataset, "updated_at": now(),
        "completed_states": states_order,
    })
    print(f"DIAGNOSTIC COMPLETE {args.dataset}: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu", choices=("0", "1"),
                        help="Expose only this physical GPU before CUDA initialization")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0,
                        help="Optional PyTorch VRAM allocator cap; 0 leaves it unset")
    parser.add_argument("--compare-schedulers", action="store_true",
                        help="Matched diagnostic recalibration with fixed LR versus official schedule")
    parser.add_argument("--limit-test", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gpu is not None:
        if torch.cuda.is_initialized():
            raise RuntimeError("Cannot select a physical GPU after CUDA initialization")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.limit_test < 2 or args.batch_size < 1 or args.workers < 0:
        parser.error("limit-test >= 2, batch-size >= 1, workers >= 0 are required")
    if not 0 <= args.cuda_memory_fraction <= 1:
        parser.error("cuda-memory-fraction must lie in [0, 1]")
    args.checkpoint = (
        args.checkpoint or ASSETS_DIR / "outputs/public5_i2cl_shot5_seed42_gpu01"
        / args.dataset / "i2cl.pt"
    ).expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if (ASSETS_DIR / "outputs").resolve() not in args.output.parents:
        raise ValueError("Use a new diagnostic directory inside radzero-assets/outputs")
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        run(args)
    except Exception:
        if args.output.exists():
            write_json(args.output / "failure.json", {
                "status": "failed", "dataset": args.dataset,
                "error": traceback.format_exc(), "failed_at": now(),
            })
        raise


if __name__ == "__main__":
    main()
