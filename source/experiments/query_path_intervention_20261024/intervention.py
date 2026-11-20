"""Frozen AlphaRAD interventions on attention and final query readout.

The two query uses are separated while preserving the author's per-head
normalization, temperature, context normalization, classification head weights,
and CLS-token handling.  ``grounding_map`` remains the author's mean scaled
cosine map.  It depends on the attention query, not on the final readout query,
and is not claimed to be a faithful attribution of the intervened classifier.

No encoder or checkpoint loading occurs here. ``vision`` is the prepared input
accepted by AlphaRAD's native ``multihead_radzero_classification_branch``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


VERSION = "alpharad-query-path-intervention-v1"
ARMS = ("original", "readout_only", "aggregation_only", "query")


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _prepare(model, vision, query):
    model = _unwrap(model)
    cls = vision["image_cls_raw_features"]
    patches = vision["image_patch_raw_features"]
    if len(cls) != 1 or len(patches) != 1:
        raise ValueError("Native AlphaRAD intervention requires one feature layer")
    cls, patches = cls[0], patches[0]
    if patches.ndim != 3 or cls.shape != (patches.shape[0], patches.shape[2]):
        raise ValueError("Expected CLS [B,D] and patches [B,P,D]")
    if patches.shape[0] == 0 or patches.shape[1] == 0:
        raise ValueError("Empty images or patches")
    if not patches.is_floating_point() or not cls.is_floating_point():
        raise ValueError("Projected visual features must be floating point")
    if cls.device != patches.device or cls.dtype != patches.dtype:
        raise ValueError("CLS and patch device/dtype must agree")
    query = torch.as_tensor(query, device=patches.device, dtype=patches.dtype)
    if query.ndim != 2 or query.shape[1] != patches.shape[-1] or not len(query):
        raise ValueError("Expected a nonempty query [C,D] matching visual features")
    if not all(torch.isfinite(x).all().item() for x in (cls, patches, query)):
        raise ValueError("Nonfinite visual features or query")
    heads = int(model.num_radzero_heads)
    if heads < 1 or patches.shape[-1] % heads:
        raise ValueError("Feature dimension must be divisible by positive head count")
    image_kv = (torch.cat([cls.unsqueeze(1), patches], dim=1)
                if model.radzero_head_enable_cross_attend_cls_token else patches)
    batch, tokens, dim = image_kv.shape
    # Keep the native operation order: split, then normalize within each head.
    keys = F.normalize(image_kv.view(batch, tokens, heads, dim // heads), p=2, dim=-1)
    query_heads = F.normalize(query.view(len(query), heads, dim // heads), p=2, dim=-1)
    scale = model.radzero_logit_scale.exp()
    if not torch.isfinite(scale).all().item():
        raise ValueError("Nonfinite native temperature scale")
    return model, keys, query_heads, scale


def _map_outputs(model, scaled_per_head, weights=None):
    result = {
        "grounding_map": scaled_per_head.mean(dim=-1),
        "grounding_per_head": scaled_per_head,
    }
    spatial = result["grounding_map"]
    if model.radzero_head_enable_cross_attend_cls_token:
        spatial = spatial[..., 1:]
    result["spatial_map_flat"] = spatial
    side = math.isqrt(spatial.shape[-1])
    if side * side == spatial.shape[-1]:
        result["similarity_scores"] = spatial.reshape(*spatial.shape[:2], side, side)
    if weights is not None:
        result["attention_weights"] = weights
        # This key follows the author's unusual naming: it is mean softmax,
        # whereas grounding_map is the mean PRE-softmax scaled cosine score.
        result["grounding_logits"] = weights.mean(dim=-1)
    return result


@torch.no_grad()
def match_intervention(model, vision, attention_query, readout_query) -> dict[str, Any]:
    """Match frozen features using independently specified query pathways.

    Returns logits [B,C], contexts [B,C,H,D/H], and native maps [B,C,K].
    K includes CLS exactly when the original model attends CLS. No parameter,
    feature, or input query is modified. Encoders must be run separately in eval
    mode; the native matching computation itself contains no dropout.
    """
    model, keys, attention_heads, scale = _prepare(model, vision, attention_query)
    readout = torch.as_tensor(readout_query, device=keys.device, dtype=keys.dtype)
    classes, heads, head_dim = attention_heads.shape
    if readout.shape != (classes, heads * head_dim) or not torch.isfinite(readout).all().item():
        raise ValueError("Readout query must be finite and match attention query shape")
    readout_heads = F.normalize(readout.view(classes, heads, head_dim), p=2, dim=-1)

    attention_logits = torch.einsum("bkhd,qhd->bqkh", keys, attention_heads)
    scaled_attention = attention_logits * scale
    attention_weights = F.softmax(scaled_attention, dim=2)
    contexts = torch.einsum("bqkh,bkhd->bqhd", attention_weights, keys)
    contexts = F.normalize(contexts, p=2, dim=-1)
    logits_per_head = torch.einsum("bqhd,qhd->bqh", contexts, readout_heads)
    logits_per_head = logits_per_head * scale

    head_weights = (torch.softmax(model.radzero_head_weights, dim=0)
                    if model.use_learnable_head_weights else None)
    mean_logits = logits_per_head.mean(-1)
    if model.head_index is not None and model.head_index >= 0:
        if model.head_index >= heads:
            raise ValueError("Selected classification head is out of range")
        logits = logits_per_head[..., model.head_index]
    elif head_weights is not None:
        if head_weights.shape != (heads,):
            raise ValueError("Native classification head weights have wrong shape")
        logits = (logits_per_head * head_weights).sum(-1)
    else:
        logits = mean_logits
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite intervened logits")
    return {
        "logits": logits,
        "logits_per_head": logits_per_head,
        "contexts": contexts,
        "class_local": F.normalize(contexts.flatten(start_dim=-2), p=2, dim=-1),
        **_map_outputs(model, scaled_attention, attention_weights),
    }


@torch.no_grad()
def evaluate_interventions(model, vision, original_query, refined_query):
    """Return Original(q/q), readout(q/q'), aggregation(q'/q), Q(q'/q')."""
    return {
        "original": match_intervention(model, vision, original_query, original_query),
        "readout_only": match_intervention(model, vision, original_query, refined_query),
        "aggregation_only": match_intervention(model, vision, refined_query, original_query),
        "query": match_intervention(model, vision, refined_query, refined_query),
    }


@torch.no_grad()
def dense_query_map(model, vision, query):
    """Reproduce the native map without attention softmax or context pooling.

    This is a diagnostic equality, not an additional spatial adaptation result.
    It demonstrates that the map can be produced from frozen dense features.
    """
    model, keys, heads, scale = _prepare(model, vision, query)
    # Matmul is intentionally independent of the matching-path einsum.
    by_head = torch.matmul(keys.permute(0, 2, 1, 3), heads.permute(1, 2, 0))
    scaled = by_head.permute(0, 3, 2, 1) * scale
    return _map_outputs(model, scaled)


@torch.no_grad()
def dense_query_residual_map(model, vision, original_query, refined_query):
    """Reproduce Q map as Original map plus a frozen dense score residual.

    The residual uses the difference of the actual per-head normalized queries.
    It preserves head-specific normalization; using a single eta-scaled global
    direction without these coefficients would not be an exact counterpart.
    """
    model, keys, original_heads, scale = _prepare(model, vision, original_query)
    _, _, refined_heads, _ = _prepare(model, vision, refined_query)
    if original_heads.shape != refined_heads.shape:
        raise ValueError("Original and refined query shapes must agree")
    baseline = dense_query_map(model, vision, original_query)["grounding_per_head"]
    delta = torch.matmul(keys.permute(0, 2, 1, 3),
                         (refined_heads - original_heads).permute(1, 2, 0))
    delta = delta.permute(0, 3, 2, 1) * scale
    return {**_map_outputs(model, baseline + delta), "dense_residual_per_head": delta}


@torch.no_grad()
def audit_native_equivalence(model, vision, original_query, refined_query,
                             *, atol=2e-6, rtol=2e-5):
    """Assert native Original/Q and dense-map equality; return error metadata.

    The audit calls the actual model's original branch, rather than treating the
    intervention implementation as its own reference. Raises on any failure.
    """
    model = _unwrap(model)
    arms = evaluate_interventions(model, vision, original_query, refined_query)
    errors = {}

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        errors[name] = float((actual - expected).abs().max().cpu())

    dtype = vision["image_patch_raw_features"][0].dtype
    device = vision["image_patch_raw_features"][0].device
    for arm, query in (("original", original_query), ("query", refined_query)):
        query = torch.as_tensor(query, device=device, dtype=dtype)
        native = model.multihead_radzero_classification_branch(
            image_embeddings=vision,
            label_embeddings={"text_cls_raw_features": query},
            generate_grounding_map=True,
        )
        for ours, reference in (("logits", "logits_label_to_image"),
                                ("logits_per_head", "logits_label_to_image_mpnce"),
                                ("contexts", "radzero_enriched_image_context"),
                                ("grounding_map", "grounding_map"),
                                ("grounding_per_head", "grounding_per_head"),
                                ("grounding_logits", "grounding_logits")):
            expected = native[reference].squeeze(-1) if ours == "logits" else native[reference]
            compare(f"{arm}.{ours}", arms[arm][ours], expected)
        dense = dense_query_map(model, vision, query)
        for key in ("grounding_map", "grounding_per_head"):
            compare(f"{arm}.dense.{key}", dense[key], arms[arm][key])
    residual = dense_query_residual_map(model, vision, original_query, refined_query)
    for key in ("grounding_map", "grounding_per_head"):
        compare(f"query.dense_residual.{key}", residual[key], arms["query"][key])
    for key in ("contexts", "grounding_map", "attention_weights"):
        compare(f"readout_fixed.{key}", arms["readout_only"][key], arms["original"][key])
        compare(f"aggregation_fixed.{key}", arms["aggregation_only"][key], arms["query"][key])
    return {"version": VERSION, "status": "passed", "atol": atol, "rtol": rtol,
            "max_absolute_errors": errors}
