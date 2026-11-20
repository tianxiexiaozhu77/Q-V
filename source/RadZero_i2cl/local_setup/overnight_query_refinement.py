#!/usr/bin/env python3
"""Support-derived query refinement through the original RadZero attention.

For each class, direction = normalize(mean(positive support class-local
vectors) - mean(negative support class-local vectors)). Nonzero eta queries
are normalize(normalize(original_query) + eta * direction). Eta zero uses the
original query unchanged. Every fixed grid member is reported; there is no
validation-based selection. Choosing a member after observing test metrics is
exploratory and is not an independently validated improvement.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from extract_frozen_features import (
    fingerprint, image_loader, public_spec, sha256, strict_baseline_check,
)
from infer_batch import MODEL_DIR, checkpoint_identity, load_model, now, write_json
from i2cl_data import load_spec
from batch_metrics import compute_metrics

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_ETAS = (0.0, 0.01, 0.03, 0.1, 0.3)
# Keep the guard aligned with the current authorized experiment window.  The
# original overnight campaign deadline was in October 2026; without updating
# it, a later reproducibility run exits before loading the cache.
# Keep the reproducibility guard ahead of the current experiment window.  The
# previous noon cutoff caused real held-out gate jobs launched later on
# 2026-11-13 to exit before reading their feature cache.
HARD_DEADLINE_TEXT = "2026-11-20T23:59:59+08:00"
HARD_DEADLINE = datetime.fromisoformat(HARD_DEADLINE_TEXT).timestamp()
VERSION = "radzero-support-query-refinement-v1"
REAGGREGATION_VERSION = "radzero-reaggregation-aware-query-refinement-v1"
ABLATION_VERSION = "radzero-fixed-point-ablation-v1"
GATE_VERSION = "radzero-support-only-pre-gate-v1"
UNCERTAINTY_VERSION = "radzero-support-uncertainty-query-refinement-v1"
JACKKNIFE_VERSION = "radzero-jackknife-stable-reaggregation-v1"
SGTT_VERSION = "radzero-support-gated-token-transport-v1"


class DeadlineReached(RuntimeError):
    pass


def parse_deadline(value):
    try:
        instant = float(value)
    except (TypeError, ValueError):
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Deadline must include its timezone")
        instant = parsed.timestamp()
    if not math.isfinite(instant) or instant > HARD_DEADLINE:
        raise ValueError(f"Deadline must not exceed {HARD_DEADLINE_TEXT}")
    return instant


def check_deadline(deadline):
    if time.time() >= deadline:
        raise DeadlineReached("Requested deadline reached")


def eta_name(eta):
    return "eta_" + format(eta, ".12g").replace(".", "p").replace("-", "m")


def validate_etas(etas):
    values = tuple(float(value) for value in etas)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Etas must be a nonempty finite nonnegative grid")
    if len(set(values)) != len(values) or len({eta_name(x) for x in values}) != len(values):
        raise ValueError("Duplicate or ambiguous eta values")
    if 0.0 not in values:
        raise ValueError("The eta grid must contain the unchanged zero-eta baseline")
    return values


def fit_query_directions(support_features, support_labels, original_queries, etas):
    """Accept support arrays only; this function has no test inputs."""
    features = torch.as_tensor(np.asarray(support_features).copy(), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or non-finite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    positive_mean = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_mean = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_mean - negative_mean
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    directions = F.normalize(difference, p=2, dim=-1)
    normalized_query = F.normalize(original, p=2, dim=-1)
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * directions, p=2, dim=-1
        ) for eta in etas
    ])
    if not torch.isfinite(queries).all():
        raise ValueError("Non-finite refined query")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(), "positive_means": positive_mean.numpy(),
        "negative_means": negative_mean.numpy(),
        "direction_norms": direction_norms.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (direction_norms < 1e-12).numpy(),
    }


@torch.no_grad()
def reaggregated_direction(support_vision, support_labels, query, matcher):
    """Compute the native RadZero support direction at ``query``.

    Cached one-shot/multi-step controls use q^(0) features for fitting, but
    their representation-gap diagnostic must re-run native attention at the
    final query.  This helper performs that explicit support-only pass.
    """
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    query = torch.as_tensor(np.asarray(query).copy(), dtype=torch.float32)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.queries.device
    features_all = []
    q = query.to(device)
    labels_d = labels.to(device)
    for vision in support_vision:
        vision = vision.to(device=device, dtype=torch.float32, non_blocking=True)
        prepared = matcher.prepare_vision(vision)
        _, attention = matcher.loss.compute_t2i_logits(q, prepared, need_attn_weights=True)
        scores = torch.stack(attention).mean(0)
        features_all.append(F.normalize(torch.matmul(
            F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1))
    features = torch.cat(features_all)
    pos = (features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
    neg = (features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
    return F.normalize(pos - neg, p=2, dim=-1).cpu().numpy()


@torch.no_grad()
def _native_support_margin(support_vision, support_labels, query, matcher):
    """Balanced native RadZero support margin for the support-only pre-gate."""
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    query = torch.as_tensor(np.asarray(query).copy(), dtype=torch.float32)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.queries.device
    labels_d, query_d = labels.to(device), query.to(device)
    logits_all = []
    for vision in support_vision:
        vision = vision.to(device=device, dtype=torch.float32, non_blocking=True)
        prepared = matcher.prepare_vision(vision)
        logits, _ = matcher.loss.compute_t2i_logits(query_d, prepared, need_attn_weights=False)
        logits_all.append(logits.reshape(len(query_d), len(vision)).T / matcher.loss.loss_temperature.exp())
    logits = torch.cat(logits_all)
    margin = (logits * labels_d).sum(0) / positives.to(device) - (
        logits * (1 - labels_d)).sum(0) / negatives.to(device)
    return margin.cpu().numpy()


@torch.no_grad()
def _native_support_observation(support_vision, support_labels, query, matcher):
    """Compute RadZero native direction and support margin in one pass."""
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    query = torch.as_tensor(np.asarray(query).copy(), dtype=torch.float32)
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    device = matcher.queries.device
    labels_d, query_d = labels.to(device), query.to(device)
    feature_batches, logits_batches = [], []
    for vision in support_vision:
        vision = vision.to(device=device, dtype=torch.float32, non_blocking=True)
        prepared = matcher.prepare_vision(vision)
        logits, attention = matcher.loss.compute_t2i_logits(
            query_d, prepared, need_attn_weights=True)
        logits = logits.reshape(len(query_d), len(vision)).T / matcher.loss.loss_temperature.exp()
        scores = torch.stack(attention).mean(0)
        feature_batches.append(F.normalize(torch.matmul(
            F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1))
        logits_batches.append(logits)
    features, logits = torch.cat(feature_batches), torch.cat(logits_batches)
    pos = (features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
    neg = (features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
    direction = F.normalize(pos - neg, p=2, dim=-1)
    margin = (logits * labels_d).sum(0) / positives.to(device) - (
        logits * (1 - labels_d)).sum(0) / negatives.to(device)
    return direction.cpu().numpy(), margin.cpu().numpy()


def fit_support_only_gate(
    support_features, support_vision, support_labels, original_queries, matcher,
    eta=0.3, gamma_margin=0.0, gamma_kappa=1.0, gamma_residual=0.0,
    max_iter=3, query_tol=1e-4, patience=1,
):
    """Pre-gate One-shot vs Full using support-only native probe evidence.

    One-shot is fitted from the cached q^(0) support representation.  A single
    native probe computes q_probe=F(q^(1)); only its support margin gain,
    direction residual, and local map ratio are used to decide whether the
    expensive fixed-point solver should run.  No test image or test label is
    touched before this decision; thresholds are explicit and must be frozen
    using an independent calibration protocol.
    """
    gate_started = time.perf_counter()
    gate_probe_started = time.perf_counter()
    if not math.isfinite(float(gamma_margin)):
        raise ValueError("gamma_margin must be finite")
    if not math.isfinite(float(gamma_kappa)) or float(gamma_kappa) < 0:
        raise ValueError("gamma_kappa must be finite and nonnegative")
    if not math.isfinite(float(gamma_residual)) or float(gamma_residual) < 0:
        raise ValueError("gamma_residual must be finite and nonnegative")
    one_shot = fit_query_directions(
        support_features, support_labels, original_queries, (0.0, float(eta)))
    q0, q1 = one_shot["queries"][0], one_shot["queries"][-1]
    d0, margin0 = _native_support_observation(support_vision, support_labels, q0, matcher)
    d1, margin1 = _native_support_observation(support_vision, support_labels, q1, matcher)
    anchor = F.normalize(torch.as_tensor(np.asarray(original_queries), dtype=torch.float32), p=2, dim=-1)
    q_probe = F.normalize(anchor + float(eta) * torch.as_tensor(d1), p=2, dim=-1).numpy()
    _, margin_probe = _native_support_observation(
        support_vision, support_labels, q_probe, matcher)
    probe_elapsed = time.perf_counter() - gate_started
    q_step = np.linalg.norm(q1 - q0, axis=-1)
    probe_step = np.linalg.norm(q_probe - q1, axis=-1)
    kappa = probe_step / np.maximum(q_step, 1e-8)
    residual = np.linalg.norm(d1 - d0, axis=-1)
    delta_margin = margin_probe - margin1
    selected_full = bool(
        float(np.mean(delta_margin)) > float(gamma_margin)
        and float(np.mean(kappa)) < float(gamma_kappa)
        and float(np.mean(residual)) >= float(gamma_residual))
    if selected_full:
        fit = fit_reaggregation_query_directions(
            support_vision, support_labels, original_queries, matcher,
            eta=eta, max_iter=max_iter, query_tol=query_tol, patience=patience)
    else:
        fit = one_shot
        gap = np.linalg.norm(d0 - d1, axis=-1)
        cosine = (d0 * d1).sum(-1)
        fit.update({"direction_initial": d0, "direction_final": d1,
                    "direction_gap_l2": gap,
                    "direction_gap_l2_mean": float(gap.mean()),
                    "direction_gap_l2_median": float(np.median(gap)),
                    "direction_gap_l2_max": float(gap.max()),
                    "direction_gap_cos": cosine,
                    "direction_gap_cos_mean": float(cosine.mean()),
                    "direction_gap_cos_min": float(cosine.min()),
                    "native_reaggregation_passes": 3})
    fit.update({
        "gate_name": "support_only_pre_gate_v1",
        "gate_selected_full": selected_full,
        "gate_thresholds": {"gamma_margin": float(gamma_margin),
                             "gamma_kappa": float(gamma_kappa),
                             "gamma_residual": float(gamma_residual)},
        "gate_probe_steps": 1,
        "gate_probe_native_passes": 3,
        "gate_probe_elapsed_seconds": float(time.perf_counter() - gate_probe_started),
        "gate_support_margin_q0": margin0,
        "gate_support_margin_q1": margin1,
        "gate_support_margin_probe": margin_probe,
        "gate_delta_margin": delta_margin,
        "gate_local_map_ratio": kappa,
        "gate_direction_residual_l2": residual,
        "gate_selected_full_rate": float(selected_full),
        "native_reaggregation_passes": (3 + int(fit.get("native_reaggregation_passes", 0))
                                        if selected_full else 3),
        "gate_total_native_passes": (3 + int(fit.get("native_reaggregation_passes", 0))
                                     if selected_full else 3),
        "estimated_fit_flops": float(
            (3 + int(fit.get("native_reaggregation_passes", 0)) if selected_full else 3)
            * len(support_labels) * np.asarray(original_queries).shape[0]
            * np.asarray(original_queries).shape[1] * 2),
        "gate_test_labels_used": False,
        "gate_pre_full_decision": True,
        # Include the native probe in timing so rejected episodes are not
        # reported as free.  Selected episodes include both probe and solver.
        "gate_probe_elapsed_seconds": float(probe_elapsed),
        "fit_elapsed_seconds": float(time.perf_counter() - gate_started),
        "wall_clock_seconds": float(time.perf_counter() - gate_started),
    })
    return fit


def fit_cached_multistep_query_directions(
    support_features, support_labels, original_queries, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1,
):
    """Cached-support multi-step ablation with no native reaggregation."""
    features = torch.as_tensor(np.asarray(support_features).copy(), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    if features.ndim != 3 or labels.shape != features.shape[:2] or original.shape != features.shape[1:]:
        raise ValueError("Expected support features [N,C,D], labels, and matching queries")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or nonfinite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("eta must be finite and positive")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid cached fixed-point iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    if not torch.allclose(torch.linalg.vector_norm(features, dim=-1),
                          torch.ones(features.shape[:2]), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    positive_means = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_means = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_means - negative_means
    directions = F.normalize(difference, p=2, dim=-1)
    direction_norms = torch.linalg.vector_norm(difference, dim=-1)
    anchor = F.normalize(original, p=2, dim=-1)
    query, trajectory = original.clone(), [original.clone()]
    history, stable_steps = [], 0
    previous_query = previous_next = None
    baseline_margin = None
    stop_reason = "max_iter"
    for iteration in range(int(max_iter)):
        normalized = F.normalize(query, p=2, dim=-1)
        projections = torch.einsum("ncd,cd->nc", features, normalized)
        margin = (projections * labels).sum(0) / positives - (projections * (1 - labels)).sum(0) / negatives
        if baseline_margin is None:
            baseline_margin = margin.clone()
        next_query = F.normalize(anchor + float(eta) * directions, p=2, dim=-1)
        delta = torch.linalg.vector_norm(next_query - normalized, dim=-1)
        if previous_query is None:
            input_delta = map_delta = local_lipschitz = None
        else:
            input_step = torch.linalg.vector_norm(normalized - previous_query, dim=-1)
            map_step = torch.linalg.vector_norm(next_query - previous_next, dim=-1)
            input_delta = float(input_step.mean())
            map_delta = float(map_step.mean())
            local_lipschitz = float((map_step / input_step.clamp_min(1e-8)).mean())
        history.append({"iteration": iteration,
                        "query_delta_max": float(delta.max()), "query_delta_mean": float(delta.mean()),
                        "support_margin_mean": float(margin.mean()), "support_margin_min": float(margin.min()),
                        "support_margin_per_class": margin.tolist(),
                        "support_utility": float((margin - baseline_margin).mean()),
                        "support_utility_per_class": (margin - baseline_margin).tolist(),
                        "input_delta_mean": input_delta, "map_delta_mean": map_delta,
                        "direction_delta_mean": 0.0 if previous_query is not None else None,
                        "local_lipschitz_mean": local_lipschitz,
                        "zero_direction_classes": int((direction_norms < 1e-12).sum()),
                        "cached_support_features": True, "native_reaggregation": False})
        previous_query, previous_next = normalized.clone(), next_query.clone()
        query = next_query
        trajectory.append(query.clone())
        stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
        if stable_steps >= int(patience):
            stop_reason = "query_stability"
            break
    gap = np.linalg.norm(directions.numpy() - directions.numpy(), axis=-1)
    return {"etas": np.asarray((0., float(eta)), dtype=np.float64),
            "original_queries": original.numpy(), "queries": np.stack((original.numpy(), query.numpy())),
            "directions": directions.numpy(), "positive_means": positive_means.numpy(),
            "negative_means": negative_means.numpy(), "direction_norms": direction_norms.numpy(),
            "positive_counts": positives.numpy().astype(np.int64), "negative_counts": negatives.numpy().astype(np.int64),
            "zero_direction_classes": (direction_norms < 1e-12).numpy(),
            "reaggregation_iterations": len(history), "reaggregation_max_iter": int(max_iter),
            "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
            "reaggregation_stop_reason": stop_reason, "reaggregation_history": history,
            "query_trajectory": torch.stack(trajectory).numpy(),
            "support_margin_baseline": baseline_margin.numpy(),
            "support_utility_final": float(history[-1]["support_utility"]),
            "local_lipschitz_max": float(max((x["local_lipschitz_mean"] for x in history
                                               if x["local_lipschitz_mean"] is not None), default=0.0)),
            "ablation_mode": "cached_multistep_no_reaggregation",
            "direction_gap_l2": gap, "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(np.median(gap)), "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos": np.ones_like(gap), "direction_gap_cos_mean": 1.0,
            "native_reaggregation_passes": 1, "fit_elapsed_seconds": 0.0}


def fit_reaggregation_only_query_directions(
    support_vision, support_labels, original_queries, matcher,
    eta=0.3, max_iter=3, query_tol=1e-4, patience=1,
):
    """Recompute native RadZero support aggregation while holding q0 fixed."""
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    if not isinstance(support_vision, (list, tuple)) or not support_vision:
        raise ValueError("support_vision must contain at least one batch")
    if labels.ndim != 2 or original.ndim != 2 or labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be finite binary values")
    if not torch.isfinite(original).all() or not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("Expected finite original query and positive eta")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid aggregation-only iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    device = matcher.queries.device
    query, labels_d = original.to(device), labels.to(device)
    history, directions = [], None
    positive_means = negative_means = direction_norms = baseline_margin = None
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            features_all, logits_all = [], []
            for vision_cpu in support_vision:
                vision = vision_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                prepared = matcher.prepare_vision(vision)
                logits, attention = matcher.loss.compute_t2i_logits(query, prepared, need_attn_weights=True)
                logits = logits.reshape(len(query), len(vision)).T / matcher.loss.loss_temperature.exp()
                scores = torch.stack(attention).mean(0)
                features_all.append(F.normalize(torch.matmul(
                    F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1))
                logits_all.append(logits)
            features, logits = torch.cat(features_all), torch.cat(logits_all)
            if features.shape[:2] != labels.shape or logits.shape != labels.shape:
                raise ValueError("Reaggregated support feature/label shape mismatch")
            positive_means = (features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
            negative_means = (features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
            difference = positive_means - negative_means
            directions = F.normalize(difference, p=2, dim=-1)
            direction_norms = torch.linalg.vector_norm(difference, dim=-1)
            margin = (logits * labels_d).sum(0) / positives.to(device) - (
                logits * (1 - labels_d)).sum(0) / negatives.to(device)
            if baseline_margin is None:
                baseline_margin = margin.clone()
            history.append({"iteration": iteration, "query_delta_max": 0.0, "query_delta_mean": 0.0,
                            "support_margin_mean": float(margin.mean()), "support_margin_min": float(margin.min()),
                            "support_margin_per_class": margin.detach().cpu().tolist(),
                            "support_utility": float((margin - baseline_margin).mean()),
                            "support_utility_per_class": (margin - baseline_margin).detach().cpu().tolist(),
                            "input_delta_mean": 0.0, "map_delta_mean": 0.0,
                            "direction_delta_mean": None, "local_lipschitz_mean": 0.0,
                            "zero_direction_classes": int((direction_norms < 1e-12).sum()),
                            "cached_support_features": False, "native_reaggregation": True})
    gap = np.linalg.norm(directions.cpu().numpy() - directions.cpu().numpy(), axis=-1)
    return {"etas": np.asarray((0., float(eta)), dtype=np.float64),
            "original_queries": original.numpy(), "queries": np.stack((original.numpy(), original.numpy())),
            "directions": directions.cpu().numpy(), "positive_means": positive_means.cpu().numpy(),
            "negative_means": negative_means.cpu().numpy(), "direction_norms": direction_norms.cpu().numpy(),
            "positive_counts": positives.numpy().astype(np.int64), "negative_counts": negatives.numpy().astype(np.int64),
            "zero_direction_classes": (direction_norms < 1e-12).cpu().numpy(),
            "reaggregation_iterations": len(history), "reaggregation_max_iter": int(max_iter),
            "reaggregation_query_tol": float(query_tol), "reaggregation_patience": int(patience),
            "reaggregation_stop_reason": "fixed_query", "reaggregation_history": history,
            "query_trajectory": np.repeat(original.numpy()[None], len(history) + 1, axis=0),
            "support_margin_baseline": baseline_margin.cpu().numpy(),
            "support_utility_final": float(history[-1]["support_utility"]),
            "local_lipschitz_max": 0.0, "ablation_mode": "reaggregation_only_fixed_query",
            "direction_gap_l2": gap, "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(np.median(gap)), "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos": np.ones_like(gap), "direction_gap_cos_mean": 1.0,
            "native_reaggregation_passes": len(history), "fit_elapsed_seconds": 0.0}


def fit_reaggregation_query_directions(support_vision, support_labels,
                                       original_queries, matcher, eta=0.3,
                                       max_iter=3, query_tol=1e-4, patience=1):
    """Recompute RadZero support attention after every query update.

    ``support_vision`` contains frozen aligned vision-token tensors.  The
    released cross-attention scorer is rerun with the current query at every
    step, so the positive/negative direction is estimated in the same
    query-conditioned representation that will score the test images.  The
    encoder and all model weights stay frozen; only support labels are used.

    The update is the fixed-point map

    ``F(q) = normalize(q_anchor + eta * d(q))``

    where ``q_anchor`` is the original text query and ``d(q)`` is recomputed
    from native RadZero attention pooled with the current query.  Keeping the
    original query as the anchor is intentional: it bounds the transport
    radius and makes the stopping criterion test convergence of the map rather
    than accumulation of an unbounded sequence of residuals.
    """
    fit_started = time.perf_counter()
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    if not isinstance(support_vision, (list, tuple)) or not support_vision:
        raise ValueError("support_vision must contain at least one batch")
    if original.ndim != 2 or labels.ndim != 2 or labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be finite binary values")
    if not torch.isfinite(original).all() or not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("Expected finite original query and positive eta")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid fixed-point iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")
    device = matcher.queries.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    labels_d = labels.to(device)
    history, stable_steps = [], 0
    # Keep the complete query path for convergence plots and estimate the
    # local Lipschitz factor of F from consecutive secants.  The estimate is
    # support-only and does not require gradients or test labels.
    query_trajectory = [F.normalize(query, p=2, dim=-1).clone()]
    previous_query = previous_map = None
    stop_reason = "max_iter"
    directions = pos = neg = direction_norms = None
    initial_directions = None
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            feature_batches, logit_batches = [], []
            for vision in support_vision:
                vision = vision.to(device=device, dtype=torch.float32, non_blocking=True)
                # RadZero's native scorer keeps the attended context implicit
                # in its logits.  Reconstruct that exact context from the
                # returned native maps (softmax(scores) @ normalized tokens)
                # so d(q) is measured in the representation actually used by
                # the scorer, rather than in a stale cached feature space.
                classes, batch = len(query), len(vision)
                prepared = matcher.prepare_vision(vision)
                logits, attention = matcher.loss.compute_t2i_logits(query, prepared, need_attn_weights=True)
                logits = logits.reshape(classes, batch).T / matcher.loss.loss_temperature.exp()
                scores = torch.stack(attention).mean(0)
                # Keep the same token set as the released pooled class-local
                # feature cache.  CLS is removed only when ``prepare_vision``
                # itself removed it; map export has separate semantics.
                pooled = F.normalize(torch.matmul(F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1)
                feature_batches.append(pooled)
                logit_batches.append(logits)
            features, logits = torch.cat(feature_batches), torch.cat(logit_batches)
            if features.shape[:2] != labels.shape or logits.shape != labels.shape:
                raise ValueError("Reaggregated support feature/label shape mismatch")
            pos = (features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
            neg = (features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
            difference = pos - neg
            direction_norms = torch.linalg.vector_norm(difference, dim=-1)
            directions = F.normalize(difference, p=2, dim=-1)
            if initial_directions is None:
                initial_directions = directions.detach().clone()
            next_query = F.normalize(anchor + float(eta) * directions, p=2, dim=-1)
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            normalized_query = F.normalize(query, p=2, dim=-1)
            if previous_query is None:
                local_lipschitz = torch.zeros_like(delta)
            else:
                input_step = torch.linalg.vector_norm(normalized_query - previous_query, dim=-1)
                output_step = torch.linalg.vector_norm(next_query - previous_map, dim=-1)
                local_lipschitz = output_step / input_step.clamp_min(1e-12)
            margin = (logits * labels_d).sum(0) / positives.to(device) - (logits * (1 - labels_d)).sum(0) / negatives.to(device)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "support_margin_mean": float(margin.mean().cpu()),
                            # ``support_utility`` is the support-only
                            # objective used by the fixed-point diagnostic:
                            # balanced positive-minus-negative native logits.
                            # Keep the explicit alias so downstream plots do
                            # not have to infer the definition from a key name.
                            "support_utility": float(margin.mean().cpu()),
                            "support_margin_min": float(margin.min().cpu()),
                            "local_lipschitz_mean": float(local_lipschitz.mean().cpu()),
                            "local_lipschitz_max": float(local_lipschitz.max().cpu()),
                            "zero_direction_classes": int((direction_norms < 1e-12).sum().cpu())})
            query_trajectory.append(next_query.clone())
            previous_query, previous_map = normalized_query.clone(), next_query.clone()
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                stop_reason = "query_stability"
                break
    # Re-evaluate the support operator at q_hat once more.  This makes the
    # exported representation gap exactly ||d(q0)-d(q_hat)|| rather than the
    # last update's d(q^(T-1)).
    with torch.no_grad():
        final_batches = []
        for vision in support_vision:
            vision = vision.to(device=device, dtype=torch.float32, non_blocking=True)
            prepared = matcher.prepare_vision(vision)
            _, attention = matcher.loss.compute_t2i_logits(query, prepared, need_attn_weights=True)
            scores = torch.stack(attention).mean(0)
            final_batches.append(F.normalize(torch.matmul(
                F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1))
        final_features = torch.cat(final_batches)
        final_pos = (final_features * labels_d[..., None]).sum(0) / positives.to(device)[:, None]
        final_neg = (final_features * (1 - labels_d)[..., None]).sum(0) / negatives.to(device)[:, None]
        final_directions = F.normalize(final_pos - final_neg, p=2, dim=-1)
    if directions is None or not torch.isfinite(query).all() or not torch.isfinite(final_directions).all():
        raise ValueError("Reaggregation produced non-finite queries")
    gap_l2 = torch.linalg.vector_norm(initial_directions - final_directions, dim=-1)
    gap_cos = (F.normalize(initial_directions, dim=-1) * F.normalize(final_directions, dim=-1)).sum(-1)
    return {"etas": np.asarray((0., float(eta)), dtype=np.float64),
            "original_queries": original.numpy(),
            "queries": np.stack((original.numpy(), query.cpu().numpy())),
            "directions": final_directions.cpu().numpy(),
            "positive_means": pos.cpu().numpy(), "negative_means": neg.cpu().numpy(),
            "direction_norms": direction_norms.cpu().numpy(),
            "positive_counts": positives.numpy().astype(np.int64),
            "negative_counts": negatives.numpy().astype(np.int64),
            "zero_direction_classes": (direction_norms < 1e-12).cpu().numpy(),
            "reaggregation_iterations": len(history),
            "reaggregation_max_iter": int(max_iter),
            "reaggregation_query_tol": float(query_tol),
            "reaggregation_patience": int(patience),
            "reaggregation_stop_reason": stop_reason,
            "reaggregation_history": history,
            "query_trajectory": torch.stack(query_trajectory).cpu().numpy(),
            "local_lipschitz_last": history[-1]["local_lipschitz_max"],
            "direction_initial": initial_directions.cpu().numpy(),
            "direction_final": final_directions.cpu().numpy(),
            "direction_gap_l2": gap_l2.cpu().numpy(),
            "direction_gap_l2_mean": float(gap_l2.mean().cpu()),
            "direction_gap_l2_median": float(gap_l2.median().cpu()),
            "direction_gap_l2_max": float(gap_l2.max().cpu()),
            "direction_gap_cos": gap_cos.cpu().numpy(),
            "direction_gap_cos_mean": float(gap_cos.mean().cpu()),
            "direction_gap_cos_min": float(gap_cos.min().cpu()),
            "native_reaggregation_passes": len(history) + 1,
            "fit_elapsed_seconds": float(time.perf_counter() - fit_started),
            "estimated_fit_flops": float((len(history) + 1) * len(labels) * original.shape[0] * original.shape[1] * 2)}


def fit_support_gated_token_transport_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, token_weight=0.5,
    alignment_threshold=0.1, support_margin_gate=False,
):
    """Support-gated token/global query transport for RadZero.

    The update is computed from the *same native similarity maps* used by
    RadZero.  A global positive-minus-negative direction is complemented by a
    token-context direction, but only when the two support-derived directions
    agree.  The agreement gate and the linear/tangent transport choice are
    fixed from support statistics, so no test labels or trainable parameters
    are introduced.  Re-running the frozen scorer after each update keeps the
    direction in the query-conditioned representation used at inference.
    """
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    if not isinstance(support_vision, (list, tuple)) or not support_vision:
        raise ValueError("support_vision must contain at least one batch")
    if labels.ndim != 2 or original.ndim != 2 or labels.shape[1] != original.shape[0]:
        raise ValueError("Support labels and queries have incompatible shapes")
    if not torch.isfinite(labels).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be finite binary values")
    if not torch.isfinite(original).all() or not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("Expected finite original query and positive eta")
    if int(max_iter) < 1 or int(max_iter) > 32 or int(patience) < 1 or int(patience) > int(max_iter):
        raise ValueError("Invalid fixed-point iteration settings")
    if not math.isfinite(float(query_tol)) or float(query_tol) < 0:
        raise ValueError("query_tol must be finite and nonnegative")
    if not math.isfinite(float(token_weight)) or float(token_weight) < 0:
        raise ValueError("token_weight must be finite and nonnegative")
    if not math.isfinite(float(alignment_threshold)) or not 0 <= float(alignment_threshold) <= 1:
        raise ValueError("alignment_threshold must lie in [0, 1]")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class needs positive and negative examples")

    device = matcher.queries.device
    anchor = F.normalize(original.to(device), p=2, dim=-1)
    query = original.to(device).clone()
    labels_d = labels.to(device)
    history, stable_steps = [], 0
    global_direction = token_direction = direction = alignment = gate = None
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            global_batches, token_batches, map_batches = [], [], []
            for vision_cpu in support_vision:
                vision = vision_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                prepared = matcher.prepare_vision(vision)
                _, attention = matcher.loss.compute_t2i_logits(
                    query, prepared, need_attn_weights=True
                )
                if attention is None or not len(attention):
                    raise ValueError("Native scorer did not return support attention maps")
                scores = torch.stack(attention).mean(0)
                if scores.ndim != 3 or scores.shape[1] != len(query):
                    raise ValueError("Unexpected RadZero attention shape")
                tokens = F.normalize(prepared, p=2, dim=-1)
                global_features = F.normalize(
                    torch.matmul(F.softmax(scores, dim=-1), tokens), p=2, dim=-1
                )
                # Contrastive token maps identify where a class's support
                # evidence differs between positives and negatives.
                map_batches.append(scores)
                global_batches.append(global_features)
                # Per-image token contexts are retained for the classwise
                # token direction below; the map contrast is applied after
                # all support images have been concatenated.
                token_batches.append(tokens)
            global_features = torch.cat(global_batches, dim=0)  # [N,C,D]
            token_values = torch.cat(token_batches, dim=0)      # [N,T,D]
            maps = torch.cat(map_batches, dim=0)                # [N,C,T]
            if global_features.shape[:2] != labels.shape:
                raise ValueError("Support attention/label shape mismatch")
            positives_d, negatives_d = positives.to(device), negatives.to(device)
            global_pos = (global_features * labels_d[..., None]).sum(0) / positives_d[:, None]
            global_neg = (global_features * (1 - labels_d)[..., None]).sum(0) / negatives_d[:, None]
            global_raw = global_pos - global_neg
            global_direction = F.normalize(global_raw, p=2, dim=-1)

            pos_map = (maps * labels_d[..., None]).sum(0) / positives_d[:, None]
            neg_map = (maps * (1 - labels_d)[..., None]).sum(0) / negatives_d[:, None]
            map_delta = pos_map - neg_map
            map_delta = F.normalize(map_delta, p=2, dim=-1)
            # A discriminative token context is built by weighting each
            # image's normalized visual tokens with the class map residual.
            token_context = torch.einsum("nct,ct,ntd->ncd", maps, map_delta, token_values)
            token_context = F.normalize(token_context, p=2, dim=-1)
            token_pos = (token_context * labels_d[..., None]).sum(0) / positives_d[:, None]
            token_neg = (token_context * (1 - labels_d)[..., None]).sum(0) / negatives_d[:, None]
            token_direction = F.normalize(token_pos - token_neg, p=2, dim=-1)

            alignment = (global_direction * token_direction).sum(-1).clamp(-1, 1)
            gate = alignment.clamp_min(0) * (alignment >= float(alignment_threshold)).to(alignment.dtype)
            step_direction = global_direction + float(token_weight) * gate[:, None] * token_direction
            step_direction = F.normalize(step_direction, p=2, dim=-1)
            linear = F.normalize(anchor + float(eta) * step_direction, p=2, dim=-1)
            tangent = step_direction - (step_direction * anchor).sum(-1, keepdim=True) * anchor
            tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
            tangent_unit = F.normalize(tangent, p=2, dim=-1)
            theta = float(eta) * tangent_norm
            tangent_query = torch.cos(theta)[:, None] * anchor + torch.sin(theta)[:, None] * tangent_unit
            # Strong agreement supports the tangent transport; ambiguous
            # classes retain the stable linear fallback used by direction 1.
            use_tangent = alignment >= float(alignment_threshold)
            next_query = torch.where(use_tangent[:, None], tangent_query, linear)
            if support_margin_gate:
                global_linear = F.normalize(anchor + float(eta) * global_direction, p=2, dim=-1)
                token_scores = torch.einsum("bcd,cd->bc", global_features, next_query)
                global_scores = torch.einsum("bcd,cd->bc", global_features, global_linear)
                token_margin = ((token_scores * labels_d).sum(0) / positives_d
                                - (token_scores * (1 - labels_d)).sum(0) / negatives_d)
                global_margin = ((global_scores * labels_d).sum(0) / positives_d
                                 - (global_scores * (1 - labels_d)).sum(0) / negatives_d)
                choose_sgtt = token_margin >= global_margin
                next_query = torch.where(choose_sgtt[:, None], next_query, global_linear)
            else:
                choose_sgtt = torch.ones_like(alignment, dtype=torch.bool)
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({
                "iteration": iteration,
                "query_delta_max": float(delta.max().cpu()),
                "query_delta_mean": float(delta.mean().cpu()),
                "support_alignment_mean": float(alignment.mean().cpu()),
                "support_gate_mean": float(gate.mean().cpu()),
                "tangent_fraction": float(use_tangent.float().mean().cpu()),
                "support_path_gate_mean": float(choose_sgtt.float().mean().cpu()),
            })
            query = next_query
            direction = step_direction
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if direction is None or not torch.isfinite(query).all():
        raise ValueError("Support-gated token transport produced non-finite queries")
    return {
        "etas": np.asarray((0., float(eta)), dtype=np.float64),
        "original_queries": original.numpy(),
        "queries": np.stack((original.numpy(), query.cpu().numpy())),
        "directions": direction.cpu().numpy(),
        "global_directions": global_direction.cpu().numpy(),
        "token_directions": token_direction.cpu().numpy(),
        "support_alignment": alignment.cpu().numpy(),
        "token_gate": gate.cpu().numpy(),
        "token_weight": float(token_weight),
        "alignment_threshold": float(alignment_threshold),
        "support_path_gate": choose_sgtt.cpu().numpy(),
        "support_margin_gate": bool(support_margin_gate),
        "geometry": "adaptive_tangent_linear",
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (torch.linalg.vector_norm(direction, dim=-1) < 1e-12).cpu().numpy(),
        "reaggregation_iterations": len(history),
        "reaggregation_max_iter": int(max_iter),
        "reaggregation_query_tol": float(query_tol),
        "reaggregation_patience": int(patience),
        "reaggregation_history": history,
    }


def _jackknife_consensus(features, labels, kappa=8.0):
    """Compute classwise leave-one-support-out consensus directions."""
    n, classes, dim = features.shape
    positive, negative = labels.sum(0), n - labels.sum(0)
    if torch.any(positive < 2) or torch.any(negative < 2):
        raise ValueError("Jackknife fitting needs at least two positives and negatives per class")
    directions = torch.zeros(classes, n, dim, dtype=features.dtype)
    valid = torch.zeros(classes, n, dtype=torch.bool)
    consensus = torch.zeros(classes, dim, dtype=features.dtype)
    stability = torch.zeros(classes, dtype=features.dtype)
    for c in range(classes):
        pos_sum = (features[:, c] * labels[:, c, None]).sum(0)
        neg_sum = (features[:, c] * (1 - labels[:, c, None])).sum(0)
        for i in range(n):
            pcount = positive[c] - labels[i, c]
            ncount = negative[c] - (1 - labels[i, c])
            if pcount < 1 or ncount < 1:
                continue
            diff = (pos_sum - labels[i, c] * features[i, c]) / pcount
            diff = diff - (neg_sum - (1 - labels[i, c]) * features[i, c]) / ncount
            norm = torch.linalg.vector_norm(diff)
            if norm > 1e-12 and torch.isfinite(norm):
                directions[c, i] = diff / norm
                valid[c, i] = True
        if not valid[c].any():
            continue
        initial = F.normalize(directions[c, valid[c]].mean(0), p=2, dim=0)
        cosine = (directions[c] * initial).sum(-1).clamp(-1, 1)
        weights = torch.where(valid[c], torch.exp(float(kappa) * cosine), torch.zeros_like(cosine))
        consensus[c] = F.normalize((weights[:, None] * directions[c]).sum(0), p=2, dim=0)
        agreement = (directions[c] * consensus[c]).sum(-1).clamp(0, 1)
        stability[c] = (agreement * valid[c]).sum() / valid[c].sum().clamp_min(1)
    return consensus, stability


def fit_jackknife_stable_query_directions(
    support_features, support_labels, original_queries, etas,
    stability_power=1.0, kappa=8.0, geometry="linear",
):
    """Frozen support-stable query transport with a jackknife trust gate."""
    features = torch.as_tensor(np.asarray(support_features).copy(), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not math.isfinite(float(stability_power)) or float(stability_power) < 0:
        raise ValueError("stability_power must be finite and nonnegative")
    if not math.isfinite(float(kappa)) or float(kappa) < 0:
        raise ValueError("kappa must be finite and nonnegative")
    if geometry not in ("linear", "tangent"):
        raise ValueError("geometry must be linear or tangent")
    if not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Non-finite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    directions, stability = _jackknife_consensus(features, labels, kappa=kappa)
    weighted = directions * stability.clamp(0, 1).pow(float(stability_power))[:, None]
    normalized_query = F.normalize(original, p=2, dim=-1)
    queries = []
    for eta in etas:
        if eta == 0:
            queries.append(original.clone())
            continue
        if geometry == "linear":
            queries.append(F.normalize(normalized_query + eta * weighted, p=2, dim=-1))
            continue
        tangent = weighted - (weighted * normalized_query).sum(-1, keepdim=True) * normalized_query
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
        tangent_unit = F.normalize(tangent, p=2, dim=-1)
        theta = float(eta) * tangent_norm.squeeze(-1)
        queries.append(torch.cos(theta)[:, None] * normalized_query
                      + torch.sin(theta)[:, None] * tangent_unit)
    queries = torch.stack(queries)
    if not torch.isfinite(queries).all():
        raise ValueError("Non-finite jackknife refined queries")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    return {"etas": np.asarray(etas, dtype=np.float64),
            "original_queries": original.numpy(), "queries": queries.numpy(),
            "directions": directions.numpy(), "weighted_directions": weighted.numpy(),
            "support_stability": stability.numpy(), "stability_power": float(stability_power),
            "jackknife_kappa": float(kappa), "geometry": geometry,
            "positive_counts": positives.numpy().astype(np.int64),
            "negative_counts": negatives.numpy().astype(np.int64),
            "zero_direction_classes": (torch.linalg.vector_norm(directions, dim=-1) < 1e-12).numpy()}


def fit_reaggregation_jackknife_query_directions(
    support_vision, support_labels, original_queries, matcher, eta=0.3,
    max_iter=3, query_tol=1e-4, patience=1, stability_power=1.0,
    kappa=8.0, geometry="tangent",
):
    """Reaggregate RadZero support tokens and transport a jackknife consensus."""
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    if not support_vision or labels.ndim != 2 or original.ndim != 2:
        raise ValueError("Invalid support vision, labels, or queries")
    if labels.shape[1] != original.shape[0] or geometry not in ("linear", "tangent"):
        raise ValueError("Invalid support/query shape or geometry")
    if eta <= 0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be finite and positive")
    if max_iter < 1 or max_iter > 32 or patience < 1 or patience > max_iter:
        raise ValueError("Invalid fixed-point iteration settings")
    positives, negatives = labels.sum(0), len(labels) - labels.sum(0)
    if torch.any(positives < 2) or torch.any(negatives < 2):
        raise ValueError("Jackknife fitting needs at least two positives and negatives")
    device = matcher.queries.device
    anchor, query = F.normalize(original.to(device), p=2, dim=-1), original.to(device).clone()
    labels_d = labels.to(device)
    history, stable_steps = [], 0
    directions = stability = None
    with torch.no_grad():
        for iteration in range(int(max_iter)):
            feature_batches = []
            for vision_cpu in support_vision:
                vision = vision_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                prepared = matcher.prepare_vision(vision)
                _, attention = matcher.loss.compute_t2i_logits(query, prepared, need_attn_weights=True)
                scores = torch.stack(attention).mean(0)
                pooled = F.normalize(torch.matmul(
                    F.softmax(scores, dim=-1), F.normalize(prepared, p=2, dim=-1)), p=2, dim=-1)
                feature_batches.append(pooled)
            features = torch.cat(feature_batches)
            directions, stability = _jackknife_consensus(features, labels_d, kappa=kappa)
            weighted = directions * stability.clamp(0, 1).pow(float(stability_power))[:, None]
            linear = F.normalize(anchor + float(eta) * weighted, p=2, dim=-1)
            if geometry == "linear":
                next_query = linear
            else:
                tangent = weighted - (weighted * anchor).sum(-1, keepdim=True) * anchor
                tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
                tangent_unit = F.normalize(tangent, p=2, dim=-1)
                theta = float(eta) * tangent_norm
                next_query = torch.cos(theta)[:, None] * anchor + torch.sin(theta)[:, None] * tangent_unit
            delta = torch.linalg.vector_norm(next_query - F.normalize(query, p=2, dim=-1), dim=-1)
            history.append({"iteration": iteration,
                            "query_delta_max": float(delta.max().cpu()),
                            "query_delta_mean": float(delta.mean().cpu()),
                            "support_stability_mean": float(stability.mean().cpu()),
                            "support_stability_min": float(stability.min().cpu())})
            query = next_query
            stable_steps = stable_steps + 1 if float(delta.max()) <= float(query_tol) else 0
            if stable_steps >= int(patience):
                break
    if directions is None:
        raise ValueError("No reaggregated support direction was produced")
    return {"etas": np.asarray((0., float(eta)), dtype=np.float64),
            "original_queries": original.numpy(),
            "queries": np.stack((original.numpy(), query.cpu().numpy())),
            "directions": directions.cpu().numpy(),
            "support_stability": stability.cpu().numpy(),
            "stability_power": float(stability_power), "jackknife_kappa": float(kappa),
            "geometry": geometry, "reaggregation_iterations": len(history),
            "reaggregation_history": history,
            "positive_counts": positives.numpy().astype(np.int64),
            "negative_counts": negatives.numpy().astype(np.int64),
            "zero_direction_classes": (torch.linalg.vector_norm(directions, dim=-1) < 1e-12).cpu().numpy()}


def fit_uncertainty_query_directions(
    support_features,
    support_labels,
    original_queries,
    etas,
    covariance_shrinkage=0.2,
    ridge=1e-3,
    reliability_power=1.0,
):
    """Support-only Mahalanobis query update with covariance shrinkage.

    The positive-minus-negative support mean is whitened by a pooled
    within-class covariance. Five-shot covariance estimates are singular, so
    the empirical matrix is shrunk toward its trace-scaled identity target and
    given a small target-scaled ridge before the Cholesky solve. A standard
    error of the mean difference yields a support reliability in [0, 1], which
    scales the update magnitude. Model weights and all test data stay frozen.
    """
    features = torch.as_tensor(np.asarray(support_features).copy(), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(support_labels).copy(), dtype=torch.float32)
    original = torch.as_tensor(np.asarray(original_queries).copy(), dtype=torch.float32)
    etas = validate_etas(etas)
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("Expected support features [N,C,D] and labels [N,C]")
    if original.shape != features.shape[1:]:
        raise ValueError("Query and support dimensions differ")
    if not len(features) or not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Empty or non-finite support/query features")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Support labels must be binary")
    if not math.isfinite(float(covariance_shrinkage)) or not 0 <= float(covariance_shrinkage) <= 1:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if not math.isfinite(float(ridge)) or float(ridge) < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if not math.isfinite(float(reliability_power)) or float(reliability_power) < 0:
        raise ValueError("reliability_power must be finite and nonnegative")
    norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-5, rtol=1e-5):
        raise ValueError("Class-local support features must have unit vector norm")
    positives = labels.sum(0)
    negatives = len(labels) - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Each support class requires positive and negative support examples")

    n_classes, dim = features.shape[1], features.shape[2]
    positive_mean = (features * labels[..., None]).sum(0) / positives[:, None]
    negative_mean = (features * (1 - labels)[..., None]).sum(0) / negatives[:, None]
    difference = positive_mean - negative_mean
    raw_norms = torch.linalg.vector_norm(difference, dim=-1)
    directions = torch.empty_like(difference)
    weighted_directions = torch.empty_like(difference)
    reliabilities = torch.empty(n_classes, dtype=features.dtype)
    mahalanobis_norms = torch.empty(n_classes, dtype=features.dtype)
    covariance_traces = torch.empty(n_classes, dtype=features.dtype)
    target_scales = torch.empty(n_classes, dtype=features.dtype)
    eye = torch.eye(dim, dtype=features.dtype)
    for class_idx in range(n_classes):
        positive = features[labels[:, class_idx] == 1, class_idx]
        negative = features[labels[:, class_idx] == 0, class_idx]
        residuals = torch.cat((positive - positive_mean[class_idx],
                               negative - negative_mean[class_idx]), dim=0)
        covariance = residuals.transpose(0, 1).matmul(residuals) / float(max(int(residuals.shape[0]) - 2, 1))
        trace = torch.trace(covariance)
        scale = torch.clamp(trace / float(dim), min=torch.finfo(features.dtype).eps)
        covariance_traces[class_idx] = trace
        target_scales[class_idx] = scale
        covariance = ((1. - float(covariance_shrinkage)) * covariance
                      + (float(covariance_shrinkage) + float(ridge)) * scale * eye)
        try:
            chol = torch.linalg.cholesky(covariance)
            whitened = torch.cholesky_solve(difference[class_idx, :, None], chol).squeeze(-1)
        except RuntimeError as exc:
            raise ValueError("Covariance regularization produced a singular matrix") from exc
        mahalanobis_norms[class_idx] = torch.sqrt(torch.dot(difference[class_idx], whitened).clamp_min(0))
        directions[class_idx] = F.normalize(whitened, p=2, dim=0)
        projected_var = torch.dot(directions[class_idx], covariance.matmul(directions[class_idx])).clamp_min(0)
        standard_error = torch.sqrt(projected_var * (1. / positives[class_idx] + 1. / negatives[class_idx]))
        reliabilities[class_idx] = (raw_norms[class_idx] /
                                    (raw_norms[class_idx] + standard_error + 1e-8)).clamp(0, 1).pow(float(reliability_power))
        weighted_directions[class_idx] = directions[class_idx] * reliabilities[class_idx]

    normalized_query = F.normalize(original, p=2, dim=-1)
    queries = torch.stack([
        original.clone() if eta == 0 else F.normalize(
            normalized_query + eta * weighted_directions, p=2, dim=-1)
        for eta in etas
    ])
    if not torch.isfinite(queries).all():
        raise ValueError("Non-finite uncertainty-aware refined query")
    return {
        "etas": np.asarray(etas, dtype=np.float64),
        "original_queries": original.numpy(), "queries": queries.numpy(),
        "directions": directions.numpy(), "weighted_directions": weighted_directions.numpy(),
        "positive_means": positive_mean.numpy(), "negative_means": negative_mean.numpy(),
        "direction_norms": raw_norms.numpy(), "mahalanobis_norms": mahalanobis_norms.numpy(),
        "support_reliability": reliabilities.numpy(), "covariance_trace": covariance_traces.numpy(),
        "covariance_target_scale": target_scales.numpy(),
        "positive_counts": positives.numpy().astype(np.int64),
        "negative_counts": negatives.numpy().astype(np.int64),
        "zero_direction_classes": (raw_norms < 1e-12).numpy(),
        "covariance_shrinkage": float(covariance_shrinkage),
        "ridge": float(ridge), "reliability_power": float(reliability_power),
    }


class QueryRefinementMatcher(nn.Module):
    """One frozen visual forward, then original attention for each fixed query.

    ``readout_only`` is an explicit diagnostic arm.  For a nonzero eta it
    keeps the original query for the attention weights (visual aggregation)
    and uses the support-updated query only for the final cosine readout.  It
    is mathematically distinct from the ordinary one-step arm, which uses the
    updated query for both operations.
    """

    def __init__(self, model, queries, etas, *, readout_only: bool = False):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.loss = model.loss_fns["KeyPhraseAlignmentLoss"]
        if self.loss.sim_op != "cos":
            raise ValueError("Query refinement expects the released cosine scorer")
        self.etas = validate_etas(etas)
        self.readout_only = bool(readout_only)
        device = next(model.parameters()).device
        values = torch.as_tensor(queries, dtype=torch.float32, device=device).clone()
        if values.ndim != 3 or len(values) != len(self.etas):
            raise ValueError("Expected queries [eta,class,hidden]")
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite query features")
        self.register_buffer("queries", values)

    def prepare_vision(self, aligned):
        vision = self.loss.layer_norm(aligned) if self.loss.layer_norm is not None else aligned
        return vision if self.loss.use_vision_cls_token else vision[:, 1:]

    def match_prepared(self, vision, query, return_maps=False):
        """Original scorer; optional raw maps correspond to the refined query."""
        classes, batch = len(query), len(vision)
        logits, attention = self.loss.compute_t2i_logits(
            query, vision, need_attn_weights=return_maps
        )
        logits = logits.reshape(classes, batch).T / self.loss.loss_temperature.exp()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite refined-query logits")
        output = {"logits": logits}
        if return_maps:
            if attention is None or not len(attention):
                raise ValueError("Missing original similarity scores")
            scores = torch.stack(attention).mean(0)
            if self.loss.use_vision_cls_token:
                scores = scores[:, :, 1:]
            side = math.isqrt(scores.shape[-1])
            if side * side != scores.shape[-1]:
                raise ValueError("Non-square patch map")
            maps = scores.reshape(batch, classes, side, side)
            if not torch.isfinite(maps).all():
                raise ValueError("Non-finite query-conditioned similarity map")
            output["similarity_scores"] = maps
        return output

    def match_readout_only_prepared(self, vision, aggregation_query,
                                    readout_query):
        """Score with q_readout after aggregating tokens with q_aggregation.

        This mirrors ``SimilarityLogit.forward`` exactly, except that the
        query used for the attention softmax is separate from the query used
        for the final cosine readout.  Keeping this operation in the native
        RadZero scorer avoids approximating readout-only QCA from saved logits.
        """
        classes, batch = len(readout_query), len(vision)
        q_agg = F.normalize(aggregation_query, p=2, dim=-1)
        q_readout = F.normalize(readout_query, p=2, dim=-1)
        local_tokens = F.normalize(vision, p=2, dim=-1)
        q_batched = q_agg.unsqueeze(0).expand(batch, classes, q_agg.shape[-1])
        q_readout_batched = q_readout.unsqueeze(0).expand(
            batch, classes, q_readout.shape[-1])
        temperature = (self.loss.attn_temperature.exp()
                       if self.loss.attn_temperature is not None
                       else self.loss.loss_temperature.exp())
        scores = torch.bmm(q_batched, local_tokens.transpose(1, 2)) / temperature
        attention = F.softmax(scores, dim=-1)
        aggregated = torch.bmm(attention, local_tokens)
        aggregated = F.normalize(aggregated, p=2, dim=-1)
        logits = torch.matmul(q_readout_batched.unsqueeze(2),
                              aggregated.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        # ``SimilarityLogit`` transposes to [classes,batch] internally and
        # ``match_prepared`` transposes it back.  This helper stays in the
        # public matcher convention [batch,classes].
        logits = logits / self.loss.loss_temperature.exp()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite readout-only logits")
        return {"logits": logits}

    @torch.no_grad()
    def forward(self, pixels, return_maps=False, deadline=None):
        self.model.eval()
        if deadline is not None:
            check_deadline(deadline)
        aligned = self.model.forward_vision_model(pixels)["vision_tokens"]
        vision = self.prepare_vision(aligned)
        outputs = {}
        for eta, query in zip(self.etas, self.queries):
            if deadline is not None:
                check_deadline(deadline)
            if self.readout_only and float(eta) != 0.0:
                outputs[eta] = self.match_readout_only_prepared(
                    vision, self.queries[0], query)
            else:
                outputs[eta] = self.match_prepared(vision, query, return_maps=return_maps)
        return outputs


def verified_cache(cache):
    """Load checked support features/labels and unlabelled test identifiers."""
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Input feature cache is incomplete")
    result = {"manifest": manifest}
    for split in ("support", "test"):
        path = cache / f"{split}_metadata.json"
        expected = manifest["auxiliary_files"][path.name]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"{split} metadata checksum differs")
        metadata = json.loads(path.read_text())
        if fingerprint(metadata) != manifest[f"{split}_identity"]:
            raise ValueError(f"{split} metadata identity differs")
        if "labels" in metadata:
            raise ValueError("Feature metadata unexpectedly contains labels")
        result[f"{split}_metadata"] = metadata
    for split, name in (("support", "class_local"), ("test", "base_logits")):
        record = manifest["splits"][split]["files"][name]
        path = cache / record["file"]
        if path.parent != cache or sha256(path) != record["sha256"]:
            raise ValueError(f"{split} {name} checksum/path differs")
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
            raise ValueError(f"{split} {name} shape/dtype differs")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite cached features/logits")
        result[f"{split}_{name}"] = values
    for name in ("support_labels", "text_features"):
        path = cache / f"{name}.npy"
        if sha256(path) != manifest["auxiliary_files"][path.name]["sha256"]:
            raise ValueError(f"{name} checksum differs")
        result[name] = np.load(path, allow_pickle=False)
    for path, digest in manifest["split_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Original split changed: {path}")
    if manifest["model_config_sha256"] != sha256(MODEL_DIR / "config.json"):
        raise ValueError("Model configuration differs from cache")
    if manifest["preprocessor_sha256"] != sha256(MODEL_DIR / "preprocessor_config.json"):
        raise ValueError("Preprocessor differs from cache")
    return result


def save_predictions(path, spec, logits):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        classes = spec["class_names"]
        writer.writerow(
            ["index", "image_path"] + [f"label:{c}" for c in classes] +
            [f"logit:{c}" for c in classes] + [f"score:{c}" for c in classes]
        )
        probabilities = torch.sigmoid(torch.from_numpy(np.asarray(logits)).double()).numpy()
        for index, image_path in enumerate(spec["image_paths"]):
            writer.writerow([index, image_path] + spec["labels"][index].tolist() +
                            logits[index].tolist() + probabilities[index].tolist())


def run(args):
    run_started = time.perf_counter()
    check_deadline(args.deadline)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    cached = verified_cache(args.cache)
    cache_manifest = cached["manifest"]
    dataset = cache_manifest["dataset"]
    args.dataset = dataset
    split_dir = Path(cache_manifest["split_dir"])
    support_spec = load_spec(split_dir, dataset, split="support")
    if public_spec(support_spec) != cached["support_metadata"]:
        raise ValueError("Cached support identity differs from original split")
    np.testing.assert_array_equal(support_spec["labels"], cached["support_labels"])
    support_metadata, test_metadata = cached["support_metadata"], cached["test_metadata"]
    for field in ("image_paths", "ids", "groups"):
        if set(support_metadata[field]) & set(test_metadata[field]):
            raise ValueError(f"Support/test overlap: {field}")
    if support_metadata["prompts"] != test_metadata["prompts"]:
        raise ValueError("Support/test prompt mismatch")
    model = processor = None
    # Non-native cached methods do not load a model or encode support images;
    # their adaptation interval starts at the dispatch point below.  Native
    # branches reset this timer after the shared frozen support encoding.
    fit_started = time.perf_counter()
    if args.method in ("reaggregation", "reaggregation_only", "support_gate", "scqt", "sgtt", "sgttm",
                       "closed_form", "cached_multistep", "readout_only"):
        check_deadline(args.deadline)
        model, processor, _ = load_model(torch.device(args.device))
        model = model.float().requires_grad_(False).eval()
        support_matcher = QueryRefinementMatcher(model, cached["text_features"][None], (0.,))
        support_vision = []
        with torch.inference_mode():
            for pixels, _, _ in image_loader(support_metadata, processor, args):
                check_deadline(args.deadline)
                aligned = model.forward_vision_model(
                    pixels.to(args.device, dtype=torch.float32, non_blocking=True)
                )["vision_tokens"]
                support_vision.append(aligned.cpu())
        # Start the adaptation timer after model loading and frozen support
        # vision encoding.  The native reaggregation solver already measures
        # its own fit interval from the branch entry; measuring closed-form
        # one-shot from the same point keeps adaptation costs comparable.
        fit_started = time.perf_counter()
        if args.method in ("closed_form", "readout_only"):
            # Original one-shot transport: use the cached native support
            # class-local features once, then perform a separate native pass
            # below for the representation-gap diagnostic.
            fit = fit_query_directions(
                cached["support_class_local"], cached["support_labels"],
                cached["text_features"], args.etas)
        elif args.method == "cached_multistep":
            fit = fit_cached_multistep_query_directions(
                cached["support_class_local"], cached["support_labels"],
                cached["text_features"], eta=args.etas[-1],
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience)
        elif args.method == "reaggregation_only":
            fit = fit_reaggregation_only_query_directions(
                support_vision, cached["support_labels"], cached["text_features"],
                support_matcher, eta=args.etas[-1],
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience)
        elif args.method == "support_gate":
            fit = fit_support_only_gate(
                cached["support_class_local"], support_vision,
                cached["support_labels"], cached["text_features"], support_matcher,
                eta=args.etas[-1], gamma_margin=args.gate_gamma_margin,
                gamma_kappa=args.gate_gamma_kappa,
                gamma_residual=args.gate_gamma_residual,
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience)
        elif args.method == "reaggregation":
            fit = fit_reaggregation_query_directions(
                support_vision, cached["support_labels"], cached["text_features"],
                support_matcher, eta=args.etas[-1],
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience)
        elif args.method == "scqt":
            fit = fit_reaggregation_jackknife_query_directions(
                support_vision, cached["support_labels"], cached["text_features"],
                support_matcher, eta=args.etas[-1],
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience,
                stability_power=args.jackknife_stability_power,
                kappa=args.jackknife_kappa, geometry=args.scqt_geometry)
        else:
            fit = fit_support_gated_token_transport_query_directions(
                support_vision, cached["support_labels"], cached["text_features"],
                support_matcher, eta=args.etas[-1],
                max_iter=args.reaggregation_max_iter,
                query_tol=args.reaggregation_query_tol,
                patience=args.reaggregation_patience,
                token_weight=args.token_weight,
                alignment_threshold=args.alignment_threshold,
                support_margin_gate=(args.method == "sgttm"))
        # These fixed-point rules return exactly the unchanged and adapted paths.
        args.etas = (0.0, float(args.etas[-1]))
    elif args.method == "uncertainty":
        fit = fit_uncertainty_query_directions(
            cached["support_class_local"], cached["support_labels"],
            cached["text_features"], args.etas,
            covariance_shrinkage=args.covariance_shrinkage,
            ridge=args.covariance_ridge,
            reliability_power=args.reliability_power,
        )
    elif args.method in ("jackknife", "tangent_jackknife"):
        fit = fit_jackknife_stable_query_directions(
            cached["support_class_local"], cached["support_labels"],
            cached["text_features"], args.etas,
            stability_power=args.jackknife_stability_power,
            kappa=args.jackknife_kappa,
            geometry="tangent" if args.method == "tangent_jackknife" else "linear")
    else:
        fit = fit_query_directions(
            cached["support_class_local"], cached["support_labels"],
            cached["text_features"], args.etas,
        )
    # Cached controls must report a true native d(q_hat) diagnostic.  Their
    # fitting rule intentionally reuses q^(0) support features, so one fresh
    # native support pass is required before declaring the gap.
    if args.method in ("closed_form", "readout_only", "cached_multistep", "reaggregation_only"):
        initial_direction = np.asarray(fit["directions"], dtype=np.float32)
        final_query = np.asarray(fit["queries"][-1], dtype=np.float32)
        final_direction = reaggregated_direction(
            support_vision, cached["support_labels"], final_query, support_matcher)
        gap = np.linalg.norm(initial_direction - final_direction, axis=-1)
        cosine = (initial_direction * final_direction).sum(-1)
        fit.update({
            "direction_initial": initial_direction,
            "direction_final": final_direction,
            "direction_gap_l2": gap,
            "direction_gap_l2_mean": float(gap.mean()),
            "direction_gap_l2_median": float(np.median(gap)),
            "direction_gap_l2_max": float(gap.max()),
            "direction_gap_cos": cosine,
            "direction_gap_cos_mean": float(cosine.mean()),
            "direction_gap_cos_min": float(cosine.min()),
            "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)) + 1,
        })
    fit.setdefault("fit_elapsed_seconds", float(time.perf_counter() - fit_started))
    fit.setdefault("native_reaggregation_passes", int(fit.get("reaggregation_iterations", 1 if args.method == "closed_form" else 0)))
    fit.setdefault("estimated_fit_flops", float(len(cached["support_labels"]) * cached["text_features"].shape[0] * cached["text_features"].shape[1] * 2))
    del support_spec
    fit_path = args.output / "fit_vectors.npz"
    np.savez(fit_path, **{key: value for key, value in fit.items()
                          if isinstance(value, np.ndarray)})
    checkpoint = checkpoint_identity()
    if checkpoint != cache_manifest["checkpoint"]:
        raise ValueError("Original checkpoint differs from cached checkpoint")
    source_paths = [Path(__file__).resolve(),
                    Path(__file__).with_name("extract_frozen_features.py").resolve(),
                    Path(__file__).with_name("batch_metrics.py").resolve(),
                    Path(__file__).with_name("infer_batch.py").resolve()]
    source_paths += sorted(MODEL_DIR.glob("*.py"))
    for path in sorted(MODEL_DIR.glob("*.py")):
        if cache_manifest["source_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"Original model code changed since caching: {path}")
    manifest = {
        "version": (GATE_VERSION if args.method == "support_gate" else
                    ABLATION_VERSION if args.method in ("cached_multistep", "reaggregation_only") else
                    REAGGREGATION_VERSION if args.method == "reaggregation" else
                    UNCERTAINTY_VERSION if args.method == "uncertainty" else
                    JACKKNIFE_VERSION if args.method in ("jackknife", "tangent_jackknife", "scqt") else
                    SGTT_VERSION if args.method in ("sgtt", "sgttm") else VERSION),
        "status": "loading_model", "created_at": now(),
        "dataset": dataset, "etas": list(args.etas), "deadline": args.deadline,
        "seed": int(args.seed),
        "cache": str(args.cache), "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "cache_identity": cache_manifest["identity"], "checkpoint": checkpoint,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "split_dir": str(split_dir), "split_sha256": cache_manifest["split_sha256"],
        "support_identity": cache_manifest["support_identity"],
        "test_identity": cache_manifest["test_identity"],
        "class_names": cache_manifest["class_names"], "prompts": cache_manifest["prompts"],
        "fit_vectors_sha256": sha256(fit_path), "fit_split": "support_only",
        "test_labels_used_for_fit": False, "trainable_base_parameters": 0,
        "fit_elapsed_seconds": float(fit.get("fit_elapsed_seconds", 0.0)),
        "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)),
        "estimated_fit_flops": float(fit.get("estimated_fit_flops", 0.0)),
        "direction_gap_l2_mean": float(fit.get("direction_gap_l2_mean", 0.0)),
        "direction_gap_l2_median": float(fit.get("direction_gap_l2_median", 0.0)),
        "direction_gap_l2_max": float(fit.get("direction_gap_l2_max", 0.0)),
        "direction_gap_cos_mean": float(fit.get("direction_gap_cos_mean", 1.0)),
        "direction_gap_cos_min": float(fit.get("direction_gap_cos_min", 1.0)),
        "gate_name": fit.get("gate_name"),
        "gate_selected_full": fit.get("gate_selected_full"),
        "gate_selected_full_rate": fit.get("gate_selected_full_rate"),
        "gate_thresholds": fit.get("gate_thresholds"),
        "gate_delta_margin_mean": (float(np.mean(fit["gate_delta_margin"]))
                                    if "gate_delta_margin" in fit else None),
        "gate_local_map_ratio_mean": (float(np.mean(fit["gate_local_map_ratio"]))
                                       if "gate_local_map_ratio" in fit else None),
        "gate_direction_residual_l2_mean": (float(np.mean(fit["gate_direction_residual_l2"]))
                                             if "gate_direction_residual_l2" in fit else None),
        "gate_test_labels_used": fit.get("gate_test_labels_used"),
        "gate_pre_full_decision": fit.get("gate_pre_full_decision"),
        "gate_probe_native_passes": fit.get("gate_probe_native_passes"),
        "gate_total_native_passes": fit.get("gate_total_native_passes"),
        "gate_probe_elapsed_seconds": fit.get("gate_probe_elapsed_seconds"),
        "batch_size": args.batch_size, "workers": args.workers,
        "baseline_atol": args.baseline_atol,
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "float32; TF32 disabled", "predictions": {},
        "selection": "No OOF selection. Entire eta grid fixed before test inference.",
        "interpretation": "Exploratory grid: selecting eta from test results is test-set selection.",
        "method": ("RadZero cross-attention with original-query aggregation and support-updated readout (readout-only QCA)."
                   if args.method == "readout_only" else
                   "RadZero cross-attention with cached multi-step queries and no support reaggregation."
                   if args.method == "cached_multistep" else
                   "RadZero native support reaggregation repeated with the original query fixed."
                   if args.method == "reaggregation_only" else
                   "Original RadZero cross-attention with reaggregation-aware fixed-point queries."
                   if args.method == "reaggregation" else
                   "Original RadZero support-only pre-gate selecting one-shot or fixed-point queries."
                   if args.method == "support_gate" else
                   "Original RadZero cross-attention with support covariance-shrinkage, "
                    "reliability-weighted class queries."
                   if args.method == "uncertainty" else
                   "Original RadZero cross-attention with jackknife-stable support consensus queries."
                   if args.method in ("jackknife", "tangent_jackknife", "scqt") else
                   "Original RadZero cross-attention with support-gated token/global transport queries."
                   if args.method in ("sgtt", "sgttm") else
                   "Original RadZero cross-attention with support-refined class queries."),
        "adaptation_mode": "one_shot" if args.method == "closed_form" else args.method,
        "ablation_mode": fit.get("ablation_mode") if args.method in ("cached_multistep", "reaggregation_only") else None,
        "reaggregation": ({key: value for key, value in fit.items()
                           if key.startswith("reaggregation_")}
                          if args.method in ("reaggregation", "cached_multistep", "reaggregation_only", "support_gate") else None),
        "support_only_gate": ({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                for key, value in fit.items() if key.startswith("gate_")}
                               if args.method == "support_gate" else None),
        "uncertainty": ({"covariance_shrinkage": args.covariance_shrinkage,
                         "ridge": args.covariance_ridge,
                         "reliability_power": args.reliability_power}
                        if args.method == "uncertainty" else None),
        "jackknife": ({"stability_power": args.jackknife_stability_power,
                       "kappa": args.jackknife_kappa,
                       "geometry": fit["geometry"]}
                      if args.method in ("jackknife", "tangent_jackknife", "scqt") else None),
        "sgtt": ({"token_weight": args.token_weight,
                  "alignment_threshold": args.alignment_threshold,
                  "geometry": fit.get("geometry", "adaptive_tangent_linear")}
                 if args.method in ("sgtt", "sgttm") else None),
        "map_semantics": "Raw similarity maps use the same refined query as classification.",
        "map_export": "No bulk maps saved; QueryRefinementMatcher.forward(return_maps=True) is available.",
    }
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "progress.json", {
        "status": "loading_model", "dataset": dataset, "updated_at": now()
    })
    check_deadline(args.deadline)
    if model is None:
        model, processor, _ = load_model(torch.device(args.device))
    matcher = QueryRefinementMatcher(
        model.float(), fit["queries"], args.etas,
        readout_only=(args.method == "readout_only"))
    count, classes = len(test_metadata["image_paths"]), len(cache_manifest["class_names"])
    arrays = {}
    for eta in args.etas:
        directory = args.output / eta_name(eta)
        directory.mkdir()
        arrays[eta] = np.lib.format.open_memmap(
            directory / "logits.npy", mode="w+", dtype=np.float32, shape=(count, classes)
        )
        arrays[eta].fill(np.nan)
    offset, last_print, started = 0, 0, time.monotonic()
    inference_started = time.perf_counter()
    baseline_check = {"passed": True, "max_abs_error": 0.0, "checked_images": 0,
                      "atol": args.baseline_atol, "rtol": 0.0, "bitwise_equal": True}
    manifest["status"] = "inferring"
    write_json(args.output / "manifest.json", manifest)
    with torch.inference_mode():
        for pixels, _, indices in image_loader(test_metadata, processor, args):
            check_deadline(args.deadline)
            stop = offset + len(pixels)
            if not np.array_equal(indices.numpy(), np.arange(offset, stop)):
                raise ValueError("Inference order differs from feature cache")
            outputs = matcher(
                pixels.to(args.device, dtype=torch.float32, non_blocking=True),
                deadline=args.deadline,
            )
            for eta, output in outputs.items():
                arrays[eta][offset:stop] = output["logits"].cpu().numpy()
            check = strict_baseline_check(
                arrays[0.0][offset:stop], cached["test_base_logits"][offset:stop],
                atol=args.baseline_atol)
            baseline_check["max_abs_error"] = max(
                baseline_check["max_abs_error"], check["max_abs_error"]
            )
            baseline_check["bitwise_equal"] &= check["bitwise_equal"]
            baseline_check["checked_images"] = stop
            offset = stop
            progress = {
                "status": "inferring", "dataset": dataset, "completed_images": offset,
                "total_images": count, "eta_count": len(args.etas), "updated_at": now(),
                "images_per_second": offset / max(time.monotonic() - started, .001),
            }
            write_json(args.output / "progress.json", progress)
            if offset == len(pixels) or offset - last_print >= 500 or offset == count:
                for values in arrays.values():
                    values.flush()
                print(f"{dataset} query grid: {offset}/{count} "
                      f"({progress['images_per_second']:.2f} images/s)", flush=True)
                last_print = offset
    if offset != count:
        raise ValueError("Incomplete query-grid inference")
    for values in arrays.values():
        values.flush()
        if not np.isfinite(values).all():
            raise ValueError("Incomplete/nonfinite query-grid predictions")
    # All grid predictions are now fixed. Test labels enter only evaluation.
    check_deadline(args.deadline)
    test_spec = load_spec(split_dir, dataset, split="test")
    if public_spec(test_spec) != test_metadata:
        raise ValueError("Test split changed between inference and scoring")
    baseline_metrics = compute_metrics(test_spec, cached["test_base_logits"])
    write_json(args.output / "baseline_metrics.json", baseline_metrics)
    metric_keys = ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")
    summary = []
    for eta in args.etas:
        check_deadline(args.deadline)
        values = arrays[eta]
        metrics = compute_metrics(test_spec, values)
        delta = {key: None if metrics[key] is None or baseline_metrics[key] is None
                 else 100.0 * (metrics[key] - baseline_metrics[key]) for key in metric_keys}
        directory = args.output / eta_name(eta)
        save_predictions(directory / "predictions.csv", test_spec, values)
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "delta_pp.json", delta)
        record = {"eta": eta, "directory": directory.name, "metrics": metrics,
                  "delta_pp": delta, "logits_sha256": sha256(directory / "logits.npy"),
                  "predictions_sha256": sha256(directory / "predictions.csv")}
        manifest["predictions"][eta_name(eta)] = record
        summary.append(record)
        write_json(args.output / "manifest.json", manifest)
    manifest.update(status="complete", completed_at=now(), baseline_check=baseline_check,
                    baseline_metrics=baseline_metrics, evaluated_images=count,
                    inference_elapsed_seconds=float(time.perf_counter() - inference_started),
                    elapsed_seconds=float(time.perf_counter() - run_started), seed=int(args.seed))
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "summary.json", {
        "status": "complete", "dataset": dataset, "baseline_metrics": baseline_metrics,
        "trials": summary, "exploratory_test_selection": True,
        "seed": int(args.seed), "fit_elapsed_seconds": float(fit.get("fit_elapsed_seconds", 0.0)),
        "inference_elapsed_seconds": float(manifest.get("inference_elapsed_seconds", 0.0)),
        "elapsed_seconds": float(manifest.get("elapsed_seconds", 0.0)),
        "native_reaggregation_passes": int(fit.get("native_reaggregation_passes", 0)),
        "estimated_fit_flops": float(fit.get("estimated_fit_flops", 0.0)),
        "direction_gap_l2_mean": float(fit.get("direction_gap_l2_mean", 0.0)),
        "direction_gap_l2_median": float(fit.get("direction_gap_l2_median", 0.0)),
        "direction_gap_l2_max": float(fit.get("direction_gap_l2_max", 0.0)),
        "direction_gap_cos_mean": float(fit.get("direction_gap_cos_mean", 1.0)),
        "direction_gap_cos_min": float(fit.get("direction_gap_cos_min", 1.0)),
    })
    write_json(args.output / "progress.json", {
        "status": "complete", "dataset": dataset, "updated_at": now(), "eta_count": len(args.etas),
        "completed_images": count,
    })
    print(f"{dataset}: query grid complete: {args.output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=42,
                        help="Deterministic evaluator seed; vary for repeated support episodes.")
    result.add_argument("--etas", type=float, nargs="+", default=list(DEFAULT_ETAS))
    result.add_argument("--method", choices=("closed_form", "readout_only", "reaggregation", "cached_multistep", "reaggregation_only", "support_gate", "uncertainty", "jackknife", "tangent_jackknife", "scqt", "sgtt", "sgttm"), default="closed_form",
                        help="Support query fit; reaggregation reruns native cross-attention")
    result.add_argument("--covariance-shrinkage", type=float, default=0.2)
    result.add_argument("--covariance-ridge", type=float, default=1e-3)
    result.add_argument("--reliability-power", type=float, default=1.0)
    result.add_argument("--jackknife-stability-power", type=float, default=1.0)
    result.add_argument("--jackknife-kappa", type=float, default=8.0)
    result.add_argument("--scqt-geometry", choices=("linear", "tangent"), default="tangent")
    result.add_argument("--token-weight", type=float, default=0.5)
    result.add_argument("--alignment-threshold", type=float, default=0.1)
    result.add_argument("--reaggregation-max-iter", type=int, default=3)
    result.add_argument("--reaggregation-query-tol", type=float, default=1e-4)
    result.add_argument("--reaggregation-patience", type=int, default=1)
    result.add_argument("--gate-gamma-margin", type=float, default=0.0,
                        help="Frozen support-only probe margin threshold for support_gate")
    result.add_argument("--gate-gamma-kappa", type=float, default=1.0,
                        help="Frozen support-only local map ratio threshold for support_gate")
    result.add_argument("--gate-gamma-residual", type=float, default=0.0,
                        help="Frozen support-only direction residual threshold for support_gate")
    result.add_argument("--baseline-atol", type=float, default=1e-6,
                        help="Historical baseline comparison tolerance")
    result.add_argument("--deadline", type=parse_deadline, default=HARD_DEADLINE_TEXT)
    return result


def main():
    args = parser().parse_args()
    args.etas = validate_etas(args.etas)
    if args.method in ("reaggregation", "reaggregation_only", "support_gate", "scqt", "sgtt", "sgttm"):
        nonzero = [eta for eta in args.etas if eta > 0]
        if len(nonzero) != 1:
            raise ValueError("Reaggregation requires exactly one nonzero eta, e.g. --etas 0 0.3")
    args.cache = args.cache.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if (args.batch_size < 1 or args.workers < 0
            or not math.isfinite(args.covariance_shrinkage)
            or not 0 <= args.covariance_shrinkage <= 1
            or not math.isfinite(args.covariance_ridge) or args.covariance_ridge < 0
            or not math.isfinite(args.reliability_power) or args.reliability_power < 0
            or not math.isfinite(args.jackknife_stability_power) or args.jackknife_stability_power < 0
            or not math.isfinite(args.jackknife_kappa) or args.jackknife_kappa < 0):
        raise ValueError("batch-size must be positive and workers nonnegative")
    if (not math.isfinite(args.token_weight) or args.token_weight < 0
            or not math.isfinite(args.alignment_threshold)
            or not 0 <= args.alignment_threshold <= 1):
        raise ValueError("Invalid SGTT token weight or alignment threshold")
    if (args.reaggregation_max_iter < 1 or args.reaggregation_max_iter > 32
            or not math.isfinite(args.reaggregation_query_tol) or args.reaggregation_query_tol < 0
            or args.reaggregation_patience < 1 or args.reaggregation_patience > args.reaggregation_max_iter):
        raise ValueError("Invalid reaggregation iteration settings")
    if (not math.isfinite(args.gate_gamma_margin)
            or not math.isfinite(args.gate_gamma_kappa) or args.gate_gamma_kappa < 0
            or not math.isfinite(args.gate_gamma_residual) or args.gate_gamma_residual < 0):
        raise ValueError("Invalid support-only gate thresholds")
    if not math.isfinite(args.baseline_atol) or args.baseline_atol <= 0:
        raise ValueError("baseline-atol must be finite and positive")
    check_deadline(args.deadline)
    args.output.mkdir(parents=True, exist_ok=False)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def deadline_alarm(_signum, _frame):
        raise DeadlineReached("Hard deadline timer expired")

    signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time()))
    try:
        run(args)
    except BaseException as error:
        status = "deadline_reached" if isinstance(error, DeadlineReached) else "failed"
        failure = {"status": status, "updated_at": now(), "error": repr(error),
                   "traceback": traceback.format_exc()}
        path = args.output / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(failure)
        write_json(path, manifest)
        write_json(args.output / "progress.json", failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
