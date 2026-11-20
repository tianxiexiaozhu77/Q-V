"""Fixed support-only query candidates shared across RadZero and AlphaRAD.

There are no test inputs, optimizers, per-dataset selection, or learned network
weights. All computation is on CPU. The first two candidates preserve the raw
FP32 query and the existing RadZero global eta=0.3 rule, respectively. Additional
candidates are a declared exploratory family, not independently validated wins.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from torch.nn import functional as F


VERSION = "support-query-candidates-v1"
CANDIDATE_NAMES = (
    "original", "uniform_eta03", "uniform_eta01", "uniform_eta06", "uniform_eta10",
    "global_variance_eta03_k1", "global_loo_eta03", "head_tangent_eta03",
    "head_tangent_variance_eta03_k1", "covariate_ridge_eta03_l1",
)


def _cpu_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).clone()
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)


def _group_codes(groups, n):
    if groups is None:
        return torch.arange(n), False
    if len(groups) != n:
        raise ValueError("groups must contain one identifier per support image")
    lookup, result = {}, []
    for group in groups:
        if isinstance(group, np.generic):
            group = group.item()
        if group is None or isinstance(group, str) and not group.strip():
            raise ValueError("Missing group identifier; pass groups=None if unavailable")
        try:
            key = json.dumps(group, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Group identifiers must be finite JSON-serializable values") from error
        if key not in lookup:
            lookup[key] = len(lookup)
        result.append(lookup[key])
    return torch.tensor(result), True


def _moments(features, labels):
    """Class-conditional means; features can have [N,C,D] or [N,C,H,Dh]."""
    broadcast = (1,) * (features.ndim - 2)
    weights = labels.reshape(*labels.shape, *broadcast)
    positives, negatives = labels.sum(0), (1 - labels).sum(0)
    positive_mean = (features * weights).sum(0) / positives.reshape(-1, *broadcast)
    negative_mean = (features * (1 - weights)).sum(0) / negatives.reshape(-1, *broadcast)
    return positive_mean, negative_mean, positives, negatives


def _global_update(original, direction, eta, gates=None, preserve_zero=False):
    normalized = F.normalize(original, p=2, dim=-1)
    unit_direction = F.normalize(direction, p=2, dim=-1)
    if gates is None:
        # Keep operation order identical to RadZero fit_query_directions.
        return F.normalize(normalized + eta * unit_direction, p=2, dim=-1)
    updated = F.normalize(normalized + (eta * gates)[:, None] * unit_direction, p=2, dim=-1)
    if preserve_zero:
        unchanged = (gates == 0) | (direction.norm(dim=-1) < 1e-12)
        updated = torch.where(unchanged[:, None], original, updated)
    return updated


def _variance(features, labels, group_codes):
    """Trace variance of a positive-minus-negative mean, with group awareness.

    For independent rows this is tr(S_pos)/n_pos + tr(S_neg)/n_neg.
    Repeated groups use a cluster sandwich estimate G/(G-1) sum_g ||U_g||²,
    U_g=sum_i_in_g a_i*(x_i-mu_label_i), a_i=y_i/n_pos-(1-y_i)/n_neg.
    Fewer than two positive or negative groups is marked unidentifiable.
    """
    positive_mean, negative_mean, positives, negatives = _moments(features, labels)
    broadcast = (1,) * (features.ndim - 2)
    weights = labels.reshape(*labels.shape, *broadcast)
    positive_residual = features - positive_mean[None]
    negative_residual = features - negative_mean[None]
    unique_groups = torch.unique(group_codes)
    positive_groups = torch.stack([labels[group_codes == g].sum(0) > 0 for g in unique_groups]).sum(0)
    negative_groups = torch.stack([(1 - labels[group_codes == g]).sum(0) > 0 for g in unique_groups]).sum(0)
    estimable = (positive_groups >= 2) & (negative_groups >= 2)
    if len(unique_groups) == len(labels):
        pos_den = (positives * (positives - 1)).clamp_min(1).reshape(-1, *broadcast)
        neg_den = (negatives * (negatives - 1)).clamp_min(1).reshape(-1, *broadcast)
        variance = ((positive_residual.square() * weights).sum(0) / pos_den
                    + (negative_residual.square() * (1 - weights)).sum(0) / neg_den).sum(-1)
        estimator = "independent class sample variance divided by class count"
    else:
        influence = (weights * positive_residual / positives.reshape(1, -1, *broadcast)
                     - (1 - weights) * negative_residual / negatives.reshape(1, -1, *broadcast))
        sums = torch.stack([influence[group_codes == g].sum(0) for g in unique_groups])
        variance = sums.square().sum(0).sum(-1)
        variance = variance * len(unique_groups) / max(len(unique_groups) - 1, 1)
        estimator = "cluster sandwich trace, G/(G-1), class-centered influence"
    return variance.clamp_min(0), estimable, {
        "variance_estimator": estimator,
        "positive_group_counts": positive_groups.tolist(),
        "negative_group_counts": negative_groups.tolist(),
        "variance_estimable": estimable.tolist(),
    }


def _variance_gates(direction, variance, estimable, kappa=1.0):
    signal = direction.square().sum(-1)
    gates = (1 - kappa * variance / signal.clamp_min(1e-12)).clamp(0, 1)
    valid = estimable.reshape(-1, *((1,) * (gates.ndim - 1)))
    gates = torch.where(valid & (signal > 1e-12), gates, torch.zeros_like(gates))
    return gates, signal


def _loo_gates(features, labels, direction, group_codes):
    """Minimum nonnegative direction agreement after leaving each full group out."""
    full = F.normalize(direction, p=2, dim=-1)
    agreements, valid_masks = [], []
    for group in torch.unique(group_codes):
        keep = group_codes != group
        reduced, y = features[keep], labels[keep]
        positives, negatives = y.sum(0), (1 - y).sum(0)
        valid = (positives > 0) & (negatives > 0)
        positive_mean = (reduced * y[..., None]).sum(0) / positives.clamp_min(1)[:, None]
        negative_mean = (reduced * (1 - y)[..., None]).sum(0) / negatives.clamp_min(1)[:, None]
        difference = positive_mean - negative_mean
        agreement = (full * F.normalize(difference, p=2, dim=-1)).sum(-1).clamp(-1, 1)
        valid = valid & (difference.norm(dim=-1) > 1e-12)
        agreements.append(agreement)
        valid_masks.append(valid)
    agreements, valid = torch.stack(agreements), torch.stack(valid_masks)
    minimum = agreements.min(0).values
    gates = torch.where(valid.all(0), minimum.clamp(0, 1), torch.zeros_like(minimum))
    return gates, {"leave_unit": "whole supplied group, otherwise one support row",
        "folds": len(agreements), "minimum_direction_cosine": minimum.tolist(),
        "valid_fold_counts": valid.sum(0).tolist(),
        "fallback": "zero correction if any leave-group-out fit loses either class or its direction"}


def _head_components(features, original, num_heads):
    n, classes, dimension = features.shape
    q = original.reshape(classes, num_heads, dimension // num_heads)
    u = F.normalize(q, p=2, dim=-1)
    # The cache has global-unit contexts; restoring head units removes the
    # shared 1/sqrt(H) factor and matches the native FLaS head representation.
    z = F.normalize(features.reshape(n, classes, num_heads, dimension // num_heads), p=2, dim=-1)
    tangent = z - (z * u[None]).sum(-1, keepdim=True) * u[None]
    return q, u, tangent


def _head_update(original, q, u, direction, gates, eta=.3):
    updated = q.norm(dim=-1, keepdim=True) * F.normalize(
        u + eta * gates[..., None] * F.normalize(direction, p=2, dim=-1), p=2, dim=-1)
    unchanged = (gates == 0) | (direction.norm(dim=-1) < 1e-12) | (q.norm(dim=-1) < 1e-12)
    updated = torch.where(unchanged[..., None], q, updated)
    return updated.reshape_as(original)


def _covariate_direction(features, labels, fallback, ridge=1.0):
    """Partial ridge regression adjusts the target effect for other disease labels.

    Center nuisance labels and scale nonconstant columns to unit population SD.
    Penalize nuisance coefficients by lambda=N*ridge; leave target/intercept
    unpenalized. Solve in float64. This is a descriptive adjustment, not a causal
    effect estimate. Single-class/constant-nuisance cases use the original mean
    difference. No parameter is estimated from evaluation data.
    """
    n, classes, _ = features.shape
    output = fallback.clone()
    diagnostics = []
    y, values = labels.double(), features.double()
    for target in range(classes):
        others = [i for i in range(classes) if i != target]
        if not others:
            diagnostics.append({"class_index": target, "fallback": "single-class task", "nuisance_count": 0})
            continue
        nuisance = y[:, others] - y[:, others].mean(0)
        scale = nuisance.square().mean(0).sqrt()
        nonconstant = scale > 1e-12
        nuisance = nuisance[:, nonconstant] / scale[nonconstant]
        count = nuisance.shape[1]
        if not count:
            diagnostics.append({"class_index": target, "fallback": "no varying nuisance labels", "nuisance_count": 0})
            continue
        target_centered = y[:, target] - y[:, target].mean()
        gram = nuisance.T @ nuisance + n * ridge * torch.eye(count, dtype=torch.float64)
        residual = target_centered - nuisance @ torch.linalg.solve(gram, nuisance.T @ target_centered)
        denominator = torch.dot(residual, target_centered)
        if denominator <= 1e-12:
            diagnostics.append({"class_index": target, "fallback": "degenerate adjusted target", "nuisance_count": count})
            continue
        centered_features = values[:, target] - values[:, target].mean(0)
        effect = (residual @ centered_features) / denominator
        if not torch.isfinite(effect).all():
            raise ValueError("Nonfinite ridge-adjusted direction")
        output[target] = effect.float()
        diagnostics.append({"class_index": target, "fallback": None, "nuisance_count": count,
                            "adjusted_target_energy": float(denominator)})
    return output, diagnostics


def fit_candidates(features, labels, original_queries, num_heads=1, groups=None):
    """Return one fixed, shared family of support-derived queries.

    Inputs: float features [N,C,D], binary labels [N,C], raw text queries [C,D].
    ``groups`` is an optional length-N sequence of actual patient/exam IDs. It
    controls variance and whole-group leave-out stability, never test selection.
    Output: names[V], FP32 CPU numpy queries[V,C,D], JSON-safe metadata[V].
    """
    features, labels, original = map(_cpu_float, (features, labels, original_queries))
    if features.ndim != 3 or labels.shape != features.shape[:2] or original.shape != features.shape[1:]:
        raise ValueError("Expected features [N,C,D], labels [N,C], queries [C,D]")
    if any(size == 0 for size in features.shape):
        raise ValueError("Empty support features")
    if not torch.isfinite(features).all() or not torch.isfinite(original).all():
        raise ValueError("Nonfinite support features or original queries")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be finite and binary")
    if isinstance(num_heads, bool) or int(num_heads) != num_heads or num_heads < 1 or features.shape[-1] % int(num_heads):
        raise ValueError("num_heads must be a positive divisor of query dimension")
    num_heads = int(num_heads)
    positive_mean, negative_mean, positives, negatives = _moments(features, labels)
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Every support class requires positive and negative examples")
    codes, groups_provided = _group_codes(groups, len(features))
    difference = positive_mean - negative_mean
    common = {"version": VERSION, "support_only": True, "test_data_used": False,
        "selection_performed": False, "network_weight_updates": False,
        "support_count": len(features), "class_count": features.shape[1],
        "query_dimension": features.shape[-1], "num_heads": num_heads,
        "positive_counts": positives.int().tolist(), "negative_counts": negatives.int().tolist(),
        "groups_provided": groups_provided, "group_count": len(torch.unique(codes)),
        "direction_norms": difference.norm(dim=-1).tolist()}
    names, queries, metadata = [], [], []

    def add(name, query, **details):
        if query.shape != original.shape or not torch.isfinite(query).all():
            raise ValueError(f"Invalid query output: {name}")
        names.append(name)
        queries.append(query)
        metadata.append({**common, "name": name, **details})

    add("original", original.clone(), algorithm="unchanged raw FP32 query", eta=0.0)
    for name, eta in (("uniform_eta03", .3), ("uniform_eta01", .1),
                      ("uniform_eta06", .6), ("uniform_eta10", 1.0)):
        add(name, _global_update(original, difference, eta), eta=eta,
            algorithm="normalize(normalize(raw_query) + eta * normalize(positive_mean-negative_mean))",
            direction_space="global", source="existing RadZero query-refinement formula")
    variance, estimable, variance_info = _variance(features, labels, codes)
    gates, signal = _variance_gates(difference, variance, estimable)
    add("global_variance_eta03_k1", _global_update(original, difference, .3, gates, True),
        algorithm="global direction with positive-part signal/noise shrinkage", eta=.3, kappa=1.,
        gate_formula="max(0, 1-kappa*mean_difference_variance/squared_direction_norm)",
        gates=gates.tolist(), variance_trace=variance.tolist(), squared_direction_norm=signal.tolist(),
        **variance_info)
    gates, loo_info = _loo_gates(features, labels, difference, codes)
    add("global_loo_eta03", _global_update(original, difference, .3, gates, True),
        algorithm="global direction shrunk by minimum leave-group-out cosine agreement", eta=.3,
        gates=gates.tolist(), **loo_info)
    q, u, tangent_features = _head_components(features, original, num_heads)
    positive_head, negative_head, _, _ = _moments(tangent_features, labels)
    tangent_direction = positive_head - negative_head
    unit_gates = torch.ones_like(tangent_direction[..., 0])
    head_info = {"eta": .3, "direction_space": "per-head tangent to original unit query",
        "preserve_raw_head_norms": True, "max_angular_displacement_radians": math.atan(.3),
        "zero_query_heads_preserved": True}
    add("head_tangent_eta03", _head_update(original, q, u, tangent_direction, unit_gates),
        algorithm="per-head tangent positive-minus-negative direction", **head_info)
    variance, estimable, variance_info = _variance(tangent_features, labels, codes)
    gates, signal = _variance_gates(tangent_direction, variance, estimable)
    add("head_tangent_variance_eta03_k1", _head_update(original, q, u, tangent_direction, gates),
        algorithm="per-head tangent direction with positive-part signal/noise shrinkage", kappa=1.,
        gates=gates.tolist(), variance_trace=variance.tolist(), squared_direction_norm=signal.tolist(),
        **head_info, **variance_info)
    adjusted, diagnostics = _covariate_direction(features, labels, difference)
    add("covariate_ridge_eta03_l1", _global_update(original, adjusted, .3),
        algorithm="other-disease partial ridge-adjusted target direction; then global query residual", eta=.3,
        ridge_strength=1., ridge_penalty="N*ridge_strength on standardized nuisance labels only",
        target_and_intercept_penalized=False, causal_interpretation=False, diagnostics=diagnostics,
        adjusted_direction_norms=adjusted.norm(dim=-1).tolist())
    if tuple(names) != CANDIDATE_NAMES:
        raise AssertionError("Candidate identities/order changed")
    result = {"names": names, "queries": torch.stack(queries).numpy().astype(np.float32, copy=False),
              "metadata": metadata}
    json.dumps(metadata, allow_nan=False)
    return result
