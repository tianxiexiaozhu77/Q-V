"""Lossless post-aggregation Direct-anchor interface for BioViL-T.

Each disease feature concatenates its raw positive and negative prompt contexts.
Direct-anchor operates on that paired feature; the original branch normalization
and positive-minus-negative readout remain frozen. Its zero reference is the
existing added-aggregation control, not the original global-image classifier.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = ROOT / "local_setup/token_mlp_adapter.py"
TOKEN_SHA256 = "07bdf46456fa31ed733da288da93c846c57aaa523e89cb943e17ddae9c60445c"
KIND = "BioViL-T_paired_context_direct_anchor_v1"
BRANCH_DIM = 128
NORMALIZE = True
QUERY_ORDER = ("negative", "positive")


def _token_module():
    if hashlib.sha256(TOKEN_PATH.read_bytes()).hexdigest() != TOKEN_SHA256:
        raise RuntimeError("Frozen BioViL-T token aggregation backend changed")
    spec = importlib.util.spec_from_file_location("_biovil_t_token_for_direct_anchor", TOKEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenHead(nn.Module):
    """CPU-exportable frozen readout; gradients to input features stay enabled."""
    def __init__(self, payload):
        super().__init__()
        if payload.get("kind") != KIND or payload.get("version") != 1:
            raise ValueError("Incompatible BioViL-T Direct-anchor head payload")
        if payload.get("dim") != 2 * BRANCH_DIM or payload.get("branch_dim") != BRANCH_DIM:
            raise ValueError("Paired feature dimension changed")
        if payload.get("normalize") is not NORMALIZE or float(payload.get("scale", 0)) != 1.0:
            raise ValueError("The frozen native branch readout must not change normalization or scale")
        if tuple(payload.get("branch_order", ())) != ("positive", "negative") or tuple(payload.get("query_order", ())) != QUERY_ORDER:
            raise ValueError("Positive/negative branch ordering changed")
        positive = payload["positive"].detach().clone()
        negative = payload["negative"].detach().clone()
        if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[1] != BRANCH_DIM:
            raise ValueError("Expected matching class-by-branch_dim positive and negative queries")
        if not positive.shape[0] or positive.dtype != negative.dtype:
            raise ValueError("Invalid frozen query shape or dtype")
        if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
            raise ValueError("Nonfinite frozen text queries")
        self.dim = 2 * BRANCH_DIM
        self.branch_dim = BRANCH_DIM
        self.class_names = tuple(payload["class_names"])
        if len(self.class_names) != positive.shape[0]:
            raise ValueError("Class list and frozen query count differ")
        self.register_buffer("positive", positive)
        self.register_buffer("negative", negative)
        queries = (negative, positive) if QUERY_ORDER[0] == "negative" else (positive, negative)
        self.register_buffer("queries", torch.cat(queries, dim=0))
        self.eval().requires_grad_(False)

    def forward(self, features):
        count = len(self.class_names)
        if features.ndim != 3 or tuple(features.shape[1:]) != (count, self.dim):
            raise ValueError(f"Expected [B,{count},{self.dim}] paired disease features")
        positive, negative = features.split(self.branch_dim, dim=-1)
        branches = (negative, positive) if QUERY_ORDER[0] == "negative" else (positive, negative)
        contexts = torch.cat(branches, dim=1)
        if NORMALIZE:
            contexts = F.normalize(contexts, dim=-1)
        scores = torch.einsum("bqd,qd->bq", contexts, self.queries)
        if QUERY_ORDER[0] == "negative":
            return scores[:, count:] - scores[:, :count]
        return scores[:, :count] - scores[:, count:]

    def export_head(self):
        return {
            "kind": KIND, "version": 1, "dim": self.dim, "branch_dim": self.branch_dim,
            "positive": self.positive.detach().cpu().clone(),
            "negative": self.negative.detach().cpu().clone(),
            "scale": 1.0, "normalize": NORMALIZE,
            "class_names": list(self.class_names),
            "branch_order": ["positive", "negative"], "query_order": list(QUERY_ORDER),
        }


def load_head(payload, device="cpu"):
    return FrozenHead(payload).to(device).eval().requires_grad_(False)


class DirectBackend:
    dim = 2 * BRANCH_DIM
    token_dim = BRANCH_DIM

    def __init__(self, device="cuda"):
        self.token_backend = _token_module().TokenBackend(device=device)
        self.device = self.token_backend.device
        self.class_names = None
        self.head = None
        self._update_metadata()

    def __getattr__(self, name):
        if name == "token_backend":
            raise AttributeError(name)
        return getattr(self.token_backend, name)

    def _update_metadata(self):
        self.metadata = deepcopy(self.token_backend.metadata)
        sources = self.metadata.get("source_files", []) + [str(TOKEN_PATH), str(Path(__file__).resolve())]
        self.metadata.update({
            "dim": self.dim, "feature_dim": self.dim, "branch_dim": self.token_dim,
            "token_dim": self.token_dim, "feature_shape": "[B,C,2*branch_dim]",
            "direct_anchor_representation": "concat(raw positive-prompt aggregate, raw negative-prompt aggregate), separately for each disease",
            "direct_anchor_branch_order": ["positive", "negative"],
            "direct_anchor_query_order": list(QUERY_ORDER),
            "direct_anchor_insertion": "After the frozen existing text-guided aggregation; before the unchanged per-branch readout",
            "direct_anchor_formula_scope": "One shared down projection and one disease-specific up projection/bias operate on the concatenated positive/negative feature; no class substitution or prompt change",
            "direct_anchor_branch_normalization": "Separate L2 normalization after adaptation" if NORMALIZE else "None; preserve native raw-dot scale after adaptation",
            "zero_anchor_reference": "aggregation_only",
            "trainable": "Only caller-supplied disease bias and Direct-anchor low-rank parameters",
            "frozen": "Original encoders/projections, text queries, added cosine attention, and original branch readout",
            "head_export_kind": KIND,
            "source_files": list(dict.fromkeys(sources)),
            "token_backend_sha256": TOKEN_SHA256,
            "source_path": str(Path(__file__).resolve()),
            "parameter_count_note": "Direct-anchor dimension is twice the original token dimension because both positive and negative visual aggregates are retained.",
        })

    def prepare_classes(self, class_names):
        self.token_backend.prepare_classes(class_names)
        self.class_names = self.token_backend.class_names
        for name in ("queries", "attention_queries", "positive", "negative"):
            if hasattr(self.token_backend, name):
                setattr(self, name, getattr(self.token_backend, name))
        count = len(self.class_names)
        if QUERY_ORDER[0] == "negative":
            negative, positive = self.queries[:count], self.queries[count:]
        else:
            positive, negative = self.queries[:count], self.queries[count:]
        self.head = load_head({
            "kind": KIND, "version": 1, "dim": self.dim, "branch_dim": self.token_dim,
            "positive": positive, "negative": negative, "scale": 1.0,
            "normalize": NORMALIZE, "class_names": list(self.class_names),
            "branch_order": ["positive", "negative"], "query_order": list(QUERY_ORDER),
        }, self.device)
        self._update_metadata()
        return self

    def _check_prepared(self):
        if self.class_names is None or self.head is None:
            raise RuntimeError("Call prepare_classes before aggregate_tokens/score_features")

    def aggregate_tokens(self, tokens):
        self._check_prepared()
        if tokens.ndim != 3 or tokens.shape[1] != self.token_backend.tokens_per_image or tokens.shape[2] != self.token_dim:
            raise ValueError("Unexpected frozen projected token shape")
        tokens = tokens.to(device=self.device, dtype=self.queries.dtype)
        attention_queries = self.attention_queries if hasattr(self.token_backend, "attention_queries") else self.queries
        weights = torch.einsum("bnd,qd->bqn", F.normalize(tokens, dim=-1), attention_queries).softmax(-1)
        contexts = torch.einsum("bqn,bnd->bqd", weights, tokens)
        count = len(self.class_names)
        if QUERY_ORDER[0] == "negative":
            return torch.cat((contexts[:, count:], contexts[:, :count]), dim=-1)
        return torch.cat((contexts[:, :count], contexts[:, count:]), dim=-1)

    def score_features(self, features):
        self._check_prepared()
        return self.head(features)

    def export_head(self):
        self._check_prepared()
        return self.head.export_head()

    def release_image_encoder(self):
        self._check_prepared()
        self.token_backend.release_image_encoder()

