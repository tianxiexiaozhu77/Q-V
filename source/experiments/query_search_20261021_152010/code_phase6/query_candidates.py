"""Eight fixed original/refined-query multiview candidates, shared by both models.

Support features are from the original image view only. Extra image views never
refit queries. This isolates test-time view averaging from support-direction
changes. Parameters and view plans are fixed without evaluation labels.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import torch
from torch.nn import functional as F

from view_geometry import VIEW_SPECS, VERSION as GEOMETRY_VERSION


VERSION = "fixed-multiview-query-candidates-v1"
CANDIDATE_NAMES = (
    "original", "uniform_eta03", "tta_flip_original", "tta_flip_refined",
    "tta_center_original", "tta_center_refined", "tta_fivecrop_original", "tta_fivecrop_refined",
)
_PLANS = {
    "original": {"views": ["identity"], "query": "original"},
    "uniform_eta03": {"views": ["identity"], "query": "uniform_eta03"},
    "tta_flip_original": {"views": ["identity", "hflip"], "query": "original"},
    "tta_flip_refined": {"views": ["identity", "hflip"], "query": "uniform_eta03"},
    "tta_center_original": {"views": ["identity", "center80"], "query": "original"},
    "tta_center_refined": {"views": ["identity", "center80"], "query": "uniform_eta03"},
    "tta_fivecrop_original": {"views": ["identity", "center80", "tl80", "tr80", "bl80", "br80"], "query": "original"},
    "tta_fivecrop_refined": {"views": ["identity", "center80", "tl80", "tr80", "bl80", "br80"], "query": "uniform_eta03"},
}


def candidate_plan(name):
    if name not in _PLANS:
        raise ValueError(f"Unknown multiview candidate {name!r}")
    return deepcopy(_PLANS[name])


def _cpu_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).clone()
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32)


def fit_candidates(features, labels, original_queries, num_heads=1, groups=None, global_features=None):
    """Return fixed names/np.float32 queries[V,C,D]/metadata for eight TTA plans.

    The optional global_features argument is accepted for worker API compatibility
    and is not used. Features are original-query class-local supports [N,C,D].
    No per-dataset or per-class TTA selection is performed.
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
    positives = labels.sum(0)
    negatives = len(labels)-positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError("Every support class requires positive and negative examples")
    group_ids = None
    if groups is not None:
        if len(groups) != len(features):
            raise ValueError("groups must contain one identifier per support image")
        group_ids = []
        for group in groups:
            if isinstance(group, np.generic):
                group = group.item()
            if group is None or isinstance(group, str) and not group.strip():
                raise ValueError("Missing support group identifier")
            try:
                group_ids.append(json.loads(json.dumps(group, allow_nan=False, sort_keys=True)))
            except (ValueError, TypeError) as error:
                raise ValueError("Groups must be finite JSON-serializable IDs") from error
    # Preserve exact old RadZero FP32 order: weighted sums, global direction
    # normalization, raw eta0, then normalized query + .3*unit direction.
    positive_mean = (features*labels[..., None]).sum(0)/positives[:, None]
    negative_mean = (features*(1-labels)[..., None]).sum(0)/negatives[:, None]
    directions = F.normalize(positive_mean-negative_mean, p=2, dim=-1)
    refined = F.normalize(F.normalize(original, p=2, dim=-1)+.3*directions, p=2, dim=-1)
    values = {"original": original, "uniform_eta03": refined}
    queries = torch.stack([values[_PLANS[name]["query"]] for name in CANDIDATE_NAMES]).numpy()
    metadata = []
    for name in CANDIDATE_NAMES:
        plan = candidate_plan(name)
        metadata.append({
            "version": VERSION, "geometry_version": GEOMETRY_VERSION, "name": name,
            "support_only_query_fit": True, "test_labels_used": False, "selection_performed": False,
            "network_weight_updates": False, "support_views": ["identity"],
            "extra_view_support_refit": False, "global_features_used": False,
            "query_source": plan["query"], "eta": 0. if plan["query"] == "original" else .3,
            "views": plan["views"], "view_specs": {view: deepcopy(VIEW_SPECS[view]) for view in plan["views"]},
            "classification_aggregation": "arithmetic mean of native logits, then original downstream sigmoid",
            "spatial_aggregation": "inverse warp raw similarity to original patch centers; mean over valid view coverage",
            "cls_aggregation": "arithmetic mean independently of spatial map",
            "inverse_warp": {"mode": "bilinear", "align_corners": False, "padding_mode": "border",
                             "outside_crop_contribution": 0, "patch_grid": [37, 37]},
            "support_count": len(features), "class_count": labels.shape[1], "query_dimension": original.shape[-1],
            "num_heads": int(num_heads), "positive_counts": positives.int().tolist(),
            "negative_counts": negatives.int().tolist(), "ordered_groups": group_ids,
            "identity_control_preserved": plan["views"] == ["identity"],
            "interpretation": "fixed test-time image-view ensemble; individual native maps are averaged after geometric restoration",
        })
    return {"names": list(CANDIDATE_NAMES), "queries": queries, "metadata": metadata}
