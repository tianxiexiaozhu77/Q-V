"""Full Direct-anchor on frozen KAD disease-conditioned decoder features.

The boundary is after the original TQN's final LayerNorm and before its frozen
two-class classifier. Positive-minus-negative logits retain the native
probability interpretation while avoiding saturated-probability ranking ties.
"""
from pathlib import Path
import importlib.util

import torch
from torch import nn


FORMAT = "kad_direct_anchor_readout_v1"


def _token_backend():
    path = Path(__file__).with_name("token_mlp_adapter.py")
    spec = importlib.util.spec_from_file_location("kad_direct_token_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TokenBackend


class FrozenHead(nn.Module):
    """Released two-class linear readout; input gradients remain enabled."""
    dim = 768

    def __init__(self, class_names):
        super().__init__()
        self.class_names = tuple(class_names)
        self.classifier = nn.Sequential(nn.Linear(self.dim, 2))
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (len(self.class_names), self.dim):
            raise ValueError(f"Expected B x {len(self.class_names)} x {self.dim} class features")
        logits = self.classifier(features)
        return logits[..., 1] - logits[..., 0]


def load_head(payload, device="cpu"):
    """Restore only the frozen classifier; no image/text encoder or TQN."""
    if payload.get("format") != FORMAT or payload.get("dim") != 768:
        raise ValueError("Incompatible KAD Direct-anchor readout")
    if payload.get("score_transform") != "positive_minus_negative_logits":
        raise ValueError("Unexpected KAD readout score convention")
    head = FrozenHead(payload["class_names"])
    head.classifier.load_state_dict(payload["classifier_state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


class DirectBackend:
    dim = 768
    token_dim = 768

    def __init__(self, device="cuda"):
        self.token_backend = _token_backend()(device=device)
        self.native = self.token_backend.native
        self.model = self.token_backend.model
        self.device = self.token_backend.device
        self.class_names = None
        self.head = None
        self.metadata = dict(self.token_backend.metadata)
        self.metadata.update({
            "adapter_method": "direct_anchor",
            "adapter_insertion": "after frozen TQN decoder final LayerNorm, before original frozen two-class linear head",
            "feature_boundary": "TQN decoder output after final LayerNorm; before eval dropout and two-class classifier",
            "feature_layout": "B x C x 768 disease-conditioned decoder features",
            "feature_dim": self.dim,
            "dim": self.dim,
            "input_token_dim": self.token_dim,
            "aggregation_origin": "native",
            "aggregation": "original text-query TQN decoder with its unchanged query self-attention and cross-attention",
            "adapter_sharing": "shared low-rank down projection; disease-specific feature bias and up projection",
            "score_transform": "original frozen two-class linear head; positive logit minus negative logit",
            "frozen": "all released image/text encoders, queries, TQN, LayerNorm and classifier; eval mode",
            "trainable": "only caller's full Direct-anchor disease bias/shared down/disease-specific up",
            "source_files": list(self.token_backend.metadata["source_files"]) + [
                str(Path(__file__)),
                str(Path(__file__).parents[1] / "A3_CLIP/models/transformer_decoder.py")],
            "head_export_format": FORMAT,
            "cache_batch_size": 32,
            "train_microbatch_size": 32,
        })

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.token_backend.prepare_classes(class_names)
        self.class_names = tuple(class_names)
        self.query = self.token_backend.query
        for key in ("class_names", "query_texts"):
            self.metadata[key] = self.token_backend.metadata[key]
        if len(self.model.mlp_head) != 1 or not isinstance(self.model.mlp_head[0], nn.Linear):
            raise ValueError("Unexpected released KAD classifier architecture")
        self.head = FrozenHead(class_names).to(self.device)
        self.head.classifier.load_state_dict(self.model.mlp_head.state_dict(), strict=True)

    def encode_images(self, image_paths):
        return self.token_backend.encode_images(image_paths)

    @torch.no_grad()
    def aggregate_tokens(self, tokens):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before aggregation")
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim:
            raise ValueError("Expected projected image patch tokens B x N x 768")
        if self.model.training:
            raise RuntimeError("Frozen KAD TQN must remain in eval mode")
        memory = self.model.decoder_norm(tokens.transpose(0, 1))
        queries = self.model.decoder_norm(self.query.unsqueeze(1).repeat(1, tokens.shape[0], 1))
        features, _ = self.model.decoder(queries, memory, memory_key_padding_mask=None,
                                        pos=None, query_pos=None)
        return features.transpose(0, 1).detach()

    def score_features(self, features):
        if self.head is None:
            raise RuntimeError("Call prepare_classes before scoring")
        return self.head(features)

    def export_head(self):
        if self.head is None:
            raise RuntimeError("Call prepare_classes before exporting the head")
        return {"format": FORMAT, "model": "KAD", "dim": self.dim,
                "class_names": list(self.class_names), "boundary": self.metadata["feature_boundary"],
                "score_transform": "positive_minus_negative_logits",
                "classifier_state_dict": {k: v.detach().cpu().clone()
                                          for k, v in self.head.classifier.state_dict().items()}}

    def native_scores_from_tokens(self, tokens):
        return self.token_backend.native_scores_from_tokens(tokens)

    def score_tokens(self, tokens, token_adapter=None):
        """Independent unchanged native token path for extraction parity."""
        return self.token_backend.score_tokens(tokens, token_adapter)

    def native_scores(self, image_paths):
        return self.token_backend.native_scores(image_paths)

    def native_predictions(self, image_paths):
        return self.token_backend.native_predictions(image_paths)

    def release_image_encoder(self):
        self.token_backend.release_image_encoder()


def create_backend(device="cuda"):
    return DirectBackend(device=device)
