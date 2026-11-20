"""Separate RadZero's native query-conditioned pooling and cosine readout.

The released ``KeyPhraseAlignmentLoss`` uses the same query first to attend
normalized visual tokens, then to read the normalized pooled context.  These
two uses are separated here.  Both native temperatures, layer normalization,
CLS policy, and the scorer's repeated query normalization are preserved.

This module does not load a checkpoint, choose devices, fit queries, or access
labels. Call ``prepare_vision`` once on ``forward_vision_model(...)["vision_tokens"]``;
all matching functions thereafter expect those prepared tokens. Original and Q
must pass ``audit_native_equivalence`` against the actual loaded native scorer
before the experiment's full inference. The original model API/encoder check
can additionally use ``FrozenFeatureExtractor.verify_official_api``.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "radzero-query-path-intervention-v1"
ARMS = ("original", "readout_only", "query")


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _loss(model):
    loss = _unwrap(model).loss_fns["KeyPhraseAlignmentLoss"]
    if loss.sim_op != "cos" or loss.similarity_logit.sim_op != "cos":
        raise ValueError("RadZero path intervention requires the released cosine scorer")
    return loss


def _validate_vision(vision):
    if (not isinstance(vision, torch.Tensor) or vision.ndim != 3
            or any(n == 0 for n in vision.shape) or not vision.is_floating_point()):
        raise ValueError("Expected nonempty floating visual tokens [B,K,D]")
    if not torch.isfinite(vision).all().item():
        raise ValueError("Nonfinite visual tokens")


@torch.no_grad()
def prepare_vision(model, aligned_tokens):
    """Apply native LayerNorm and CLS inclusion exactly once to aligned tokens."""
    _validate_vision(aligned_tokens)
    loss = _loss(model)
    vision = loss.layer_norm(aligned_tokens) if loss.layer_norm is not None else aligned_tokens
    if not loss.use_vision_cls_token:
        vision = vision[:, 1:]
    _validate_vision(vision)
    return vision


def _query(query, vision, name):
    query = torch.as_tensor(query, dtype=vision.dtype, device=vision.device)
    if (query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != vision.shape[-1]
            or not torch.isfinite(query).all().item()):
        raise ValueError(f"{name} must be finite [C,D] matching visual dimension")
    return query


@torch.no_grad()
def match_intervention(model, prepared_vision, attention_query, readout_query) -> dict[str, Any]:
    """Compute native pooling with q_agg, followed by cosine readout with q_read.

    ``prepared_vision`` already includes native LayerNorm and CLS selection.
    The returned pre-softmax maps depend on q_agg and are not an attribution
    of the additional readout term. Shapes never squeeze B or C dimensions.
    """
    loss = _loss(model)
    vision = prepared_vision
    _validate_vision(vision)
    attention_query = _query(attention_query, vision, "Attention query")
    readout_query = _query(readout_query, vision, "Readout query")
    if attention_query.shape != readout_query.shape:
        raise ValueError("Attention/readout query shapes must agree")
    attention_temperature = (loss.attn_temperature.exp()
                             if loss.attn_temperature is not None
                             else loss.loss_temperature.exp())
    output_temperature = loss.loss_temperature.exp()
    for temperature in (attention_temperature, output_temperature):
        if (not torch.isfinite(temperature).all().item()
                or not (temperature > 0).all().item()):
            raise ValueError("Native temperatures must be finite and positive")

    batch, _, _ = vision.shape
    classes = len(attention_query)
    # Expand before normalization, retaining the released scorer's operation order.
    attention_features = F.normalize(attention_query.unsqueeze(0).expand(
        batch, classes, attention_query.shape[-1]), p=2, dim=-1)
    local_tokens = F.normalize(vision, p=2, dim=-1)
    scores = torch.bmm(attention_features, local_tokens.permute(0, 2, 1)) / attention_temperature
    attention_weights = F.softmax(scores, dim=-1)
    contexts = F.normalize(torch.matmul(attention_weights, local_tokens), p=2, dim=-1)
    # Native SimilarityLogit normalizes its query twice: before attention and
    # again before readout. Preserve both, including for original=q_read.
    readout_features = F.normalize(readout_query.unsqueeze(0).expand(
        batch, classes, readout_query.shape[-1]), p=2, dim=-1)
    readout_features = F.normalize(readout_features, p=2, dim=-1)
    logits = torch.matmul(readout_features.unsqueeze(2), contexts.unsqueeze(-1))
    logits = logits.reshape(batch, classes) / output_temperature
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite intervened logits")
    spatial = scores[:, :, 1:] if loss.use_vision_cls_token else scores
    result = {"logits": logits, "contexts": contexts, "class_local": contexts,
              "attention_weights": attention_weights, "grounding_map": scores,
              "spatial_map_flat": spatial}
    side = math.isqrt(spatial.shape[-1])
    if side * side == spatial.shape[-1] and side > 0:
        result["similarity_scores"] = spatial.reshape(batch, classes, side, side)
    return result


@torch.no_grad()
def evaluate_interventions(model, prepared_vision, original_query, refined_query):
    """Return Original(q/q), R(q/q'), and Q(q'/q') from one visual forward."""
    return {
        "original": match_intervention(model, prepared_vision, original_query, original_query),
        "readout_only": match_intervention(model, prepared_vision, original_query, refined_query),
        "query": match_intervention(model, prepared_vision, refined_query, refined_query),
    }


def source_provenance(model):
    """Record the actual loaded scorer code and its temperature/CLS settings."""
    loss = _loss(model)
    sources = {}
    for obj in (type(loss), type(loss.similarity_logit)):
        path = Path(inspect.getfile(obj)).resolve()
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"native_source_sha256": sources,
            "use_vision_cls_token": bool(loss.use_vision_cls_token),
            "use_layer_norm": loss.layer_norm is not None,
            "attention_temperature": float((loss.attn_temperature.exp()
                if loss.attn_temperature is not None else loss.loss_temperature.exp()).detach().cpu()),
            "output_temperature": float(loss.loss_temperature.exp().detach().cpu())}


@torch.no_grad()
def audit_native_equivalence(model, prepared_vision, original_query, refined_query,
                             *, reference_matcher=None, atol=2e-5, rtol=1e-4):
    """Assert native Original/Q plus optional existing QueryRefinementMatcher.

    References are obtained by calling the actual loaded released scorer, not
    another copy of this intervention. Singleton B/C axes are explicitly tested
    by the CPU tests and should also be included in the real-image smoke run.
    """
    loss = _loss(model)
    arms = evaluate_interventions(model, prepared_vision, original_query, refined_query)
    errors = {}

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        errors[name] = float((actual - expected).abs().max().cpu())

    for arm, query in (("original", original_query), ("query", refined_query)):
        query = _query(query, prepared_vision, arm)
        native_logits, native_maps = loss.compute_t2i_logits(
            query, prepared_vision, need_attn_weights=True)
        native_logits = native_logits.reshape(len(query), len(prepared_vision)).T
        native_logits = native_logits / loss.loss_temperature.exp()
        if native_maps is None or len(native_maps) != 1:
            raise ValueError("Expected exactly one native cosine attention map")
        compare(f"{arm}.native_logits", arms[arm]["logits"], native_logits)
        compare(f"{arm}.native_grounding_map", arms[arm]["grounding_map"], native_maps[0])
        if reference_matcher is not None:
            reference = reference_matcher.match_prepared(prepared_vision, query, return_maps=True)
            compare(f"{arm}.existing_matcher_logits", arms[arm]["logits"], reference["logits"])
            compare(f"{arm}.existing_matcher_map", arms[arm]["similarity_scores"],
                    reference["similarity_scores"])
    for key in ("contexts", "attention_weights", "grounding_map"):
        compare(f"readout_fixed.{key}", arms["readout_only"][key], arms["original"][key])
    return {"version": VERSION, "status": "passed", "atol": atol, "rtol": rtol,
            "max_absolute_errors": errors, **source_provenance(model)}


class RadZeroPathMatcher(nn.Module):
    """Convenience wrapper for one visual forward and all Original/R/Q arms."""

    def __init__(self, model, original_query, refined_query):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        _loss(self.model)
        parameter = next(self.model.parameters())
        self.register_buffer("original_query", torch.as_tensor(
            original_query, device=parameter.device, dtype=parameter.dtype).clone())
        self.register_buffer("refined_query", torch.as_tensor(
            refined_query, device=parameter.device, dtype=parameter.dtype).clone())

    def prepare_vision(self, aligned_tokens):
        return prepare_vision(self.model, aligned_tokens)

    def match_prepared(self, vision, attention_query, readout_query):
        return match_intervention(self.model, vision, attention_query, readout_query)

    @torch.no_grad()
    def forward(self, pixels):
        self.model.eval()
        aligned = self.model.forward_vision_model(pixels)["vision_tokens"]
        vision = self.prepare_vision(aligned)
        return evaluate_interventions(self.model, vision, self.original_query, self.refined_query)
