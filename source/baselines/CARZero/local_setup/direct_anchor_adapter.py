"""Post-DQN Direct-anchor boundary retaining both frozen CARZero branches.

The class feature is concat(i2t, t2i), after each original decoder's final
LayerNorm and before its frozen MLP classifier. A disease-specific residual
may mix the two 768-dimensional halves. Scoring restores the two branches and
their original arithmetic mean; no attention or classifier is retrained.
"""
from pathlib import Path
import importlib.util

import torch
from torch import nn


FORMAT = "carzero_direct_anchor_readout_v1"


def _token_backend():
    path = Path(__file__).with_name("token_mlp_adapter.py")
    spec = importlib.util.spec_from_file_location("carzero_direct_token_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TokenBackend


class FrozenHead(nn.Module):
    """Small released classifier only; differentiable with respect to input."""
    dim = 1536

    def __init__(self, class_names):
        super().__init__()
        self.class_names = tuple(class_names)
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(inplace=True), nn.Dropout(.1),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(.1),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(.1),
            nn.Linear(256, 1))
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        # Frozen author dropout must remain disabled even if a parent module
        # is switched to train mode during adapter optimization.
        return super().train(False)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (len(self.class_names), self.dim):
            raise ValueError(f"Expected B x {len(self.class_names)} x {self.dim} class features")
        i2t = self.classifier(features[..., :768]).squeeze(-1)
        t2i = self.classifier(features[..., 768:]).squeeze(-1)
        return (i2t + t2i) / 2


def load_head(payload, device="cpu"):
    """Restore only the frozen readout; no encoders, queries or DQN are loaded."""
    if payload.get("format") != FORMAT or payload.get("dim") != 1536:
        raise ValueError("Incompatible CARZero Direct-anchor readout")
    if payload.get("branch_order") != ["i2t", "t2i"]:
        raise ValueError("CARZero branch order must be i2t followed by t2i")
    head = FrozenHead(payload["class_names"])
    head.classifier.load_state_dict(payload["classifier_state_dict"], strict=True)
    return head.to(device).eval().requires_grad_(False)


class DirectBackend:
    dim = 1536
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
            "adapter_insertion": "after both frozen DQN decoders' final LayerNorm, before original frozen MLP classifiers",
            "feature_boundary": "concat of post-final-LayerNorm i2t and t2i decoder outputs; eval dropout is identity",
            "feature_layout": "B x C x 1536 = concat(i2t[768], t2i[768])",
            "feature_dim": self.dim,
            "dim": self.dim,
            "input_token_dim": self.token_dim,
            "aggregation_origin": "native",
            "aggregation": "original bidirectional DQN; preserve both directions and their independent frozen readout",
            "adapter_sharing": "shared low-rank down projection; disease-specific feature bias and up projection",
            "adapted_branches": "joint i2t+t2i class feature; low-rank residual may mix branch dimensions",
            "score_transform": "original shared nonlinear MLP on each 768-dimensional half, then arithmetic mean of logits",
            "frozen": "all released encoders, queries, DQN, LayerNorm and MLP classifiers; eval mode",
            "trainable": "only caller's full Direct-anchor disease bias/shared down/disease-specific up",
            "source_files": list(self.token_backend.metadata["source_files"]) + [
                str(Path(__file__)),
                str(Path(__file__).parents[1] / "CARZero/models/transformer_decoder.py")],
            "head_export_format": FORMAT,
            "cache_batch_size": 16,
            "train_microbatch_size": 32,
        })

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.token_backend.prepare_classes(class_names)
        self.class_names = tuple(class_names)
        self.query = self.token_backend.query
        self.text_memory = self.token_backend.text_memory
        for key in ("class_names", "last_prompts", "query_texts"):
            self.metadata[key] = self.token_backend.metadata[key]
        native_head = self.model.fusion_module.mlp_head
        expected = [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout,
                    nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
        if [type(layer) for layer in native_head] != expected:
            raise ValueError("Unexpected released CARZero classifier architecture")
        self.head = FrozenHead(class_names).to(self.device)
        self.head.classifier.load_state_dict(native_head.state_dict(), strict=True)

    def encode_images(self, image_paths):
        return self.token_backend.encode_images(image_paths)

    @staticmethod
    def _decoder_features(fusion, memory, queries):
        batch = memory.shape[0]
        memory = fusion.decoder_norm(memory.transpose(0, 1))
        queries = fusion.decoder_norm(queries.unsqueeze(1).repeat(1, batch, 1))
        features, _ = fusion.decoder(queries, memory, memory_key_padding_mask=None,
                                     pos=None, query_pos=None)
        # Original dropout_feas is identity in the required frozen eval mode.
        return features.transpose(0, 1)

    @torch.no_grad()
    def aggregate_tokens(self, tokens):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before aggregation")
        if tokens.ndim != 3 or tokens.shape[-1] != self.token_dim or tokens.shape[1] < 2:
            raise ValueError("Expected projected global+patch image tokens B x N x 768")
        fusion = self.model.fusion_module
        if fusion.training:
            raise RuntimeError("Frozen CARZero DQN must remain in eval mode")
        i2t = self._decoder_features(fusion, tokens, self.query)
        # Native reverse branch uses text memory as the batch. Its decoder
        # has no query self-attention, so the image queries are independent.
        t2i = self._decoder_features(fusion, self.text_memory, tokens[:, 0]).transpose(0, 1)
        return torch.cat([i2t, t2i], dim=-1).detach()

    def score_features(self, features):
        if self.head is None:
            raise RuntimeError("Call prepare_classes before scoring")
        return self.head(features)

    def export_head(self):
        if self.head is None:
            raise RuntimeError("Call prepare_classes before exporting the head")
        return {"format": FORMAT, "model": "CARZero", "dim": self.dim,
                "class_names": list(self.class_names), "branch_order": ["i2t", "t2i"],
                "boundary": self.metadata["feature_boundary"],
                "score_transform": self.metadata["score_transform"],
                "classifier_state_dict": {k: v.detach().cpu().clone()
                                          for k, v in self.head.classifier.state_dict().items()}}

    def native_scores_from_tokens(self, tokens):
        return self.token_backend.native_scores_from_tokens(tokens)

    def score_tokens(self, tokens, token_adapter=None):
        """Independent unchanged native token path for extraction parity."""
        return self.token_backend.score_tokens(tokens, token_adapter)

    def native_scores(self, image_paths):
        return self.token_backend.native_scores(image_paths)

    native_predictions = native_scores

    def release_image_encoder(self):
        self.token_backend.release_image_encoder()


def create_backend(device="cuda"):
    return DirectBackend(device=device)
