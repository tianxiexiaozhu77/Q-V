"""Train a residual token adapter while keeping released RadZero scoring frozen.

This module consumes ``model.forward_vision_model(pixels)["vision_tokens"]``
before the released KeyPhraseAlignmentLoss applies its layer normalization.
It does not add a classifier or change the official checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn


class TokenResidualAdapter(nn.Module):
    """An initially identity, 768 -> 32 -> 768 residual adapter by default."""

    def __init__(
        self,
        hidden_dim: int = 768,
        bottleneck: int = 32,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck <= 0:
            raise ValueError("hidden_dim and bottleneck must be positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.scale = float(scale)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_dim)
        # Down uses PyTorch's standard initialization and the caller's RNG seed.
        # Zero output starts from the released model's exact predictions.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected [batch, tokens, {self.hidden_dim}], got {tuple(tokens.shape)}"
            )
        return tokens + self.scale * self.up(self.activation(self.down(tokens)))


class RadZeroAdapterHead(nn.Module):
    """Single-prompt adapter followed by the unmodified released scoring module.

    Only ``adapter`` parameters are trainable. ``text_features`` must be the
    post-normalization features returned by the official compute_text_features.
    Features cached by a frozen encoder need no gradients, but this forward
    must run with autograd enabled when training the adapter.
    """

    def __init__(
        self,
        loss_module: nn.Module,
        text_features: Tensor,
        bottleneck: int = 32,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(loss_module.hidden_dim)
        if text_features.ndim != 2 or text_features.shape != (1, hidden_dim):
            raise ValueError(
                f"SIIM requires one text feature of shape [1, {hidden_dim}], "
                f"got {tuple(text_features.shape)}"
            )
        self.loss_module = loss_module.requires_grad_(False).eval()
        # A text feature computed in an outer inference_mode context must become
        # a normal tensor: autograd saves it while differentiating image tokens.
        with torch.inference_mode(False):
            self.register_buffer("text_features", text_features.detach().clone())
            self.adapter = TokenResidualAdapter(hidden_dim, bottleneck, scale).to(
                device=text_features.device, dtype=text_features.dtype
            )

    def train(self, mode: bool = True) -> "RadZeroAdapterHead":
        super().train(mode)
        self.loss_module.eval()
        return self

    def forward(self, tokens: Tensor, return_maps: bool = False) -> dict[str, Tensor]:
        vision = self.adapter(tokens)
        if self.loss_module.layer_norm is not None:
            vision = self.loss_module.layer_norm(vision)
        if not self.loss_module.use_vision_cls_token:
            vision = vision[:, 1:]
        logits, attention = self.loss_module.compute_t2i_logits(
            self.text_features, vision, need_attn_weights=return_maps
        )
        # The released implementation squeezes singleton dimensions. Restore
        # [class, batch] explicitly for both single-image and larger batches.
        logits = logits.reshape(1, tokens.shape[0]).T
        logits = logits / self.loss_module.loss_temperature.exp()
        output = {"logits": logits}
        if return_maps:
            if attention is None or len(attention) == 0:
                raise RuntimeError("Released scoring returned no similarity maps")
            scores = torch.stack(attention).mean(dim=0)
            if self.loss_module.use_vision_cls_token:
                scores = scores[:, :, 1:]
            side = math.isqrt(scores.shape[-1])
            if side * side != scores.shape[-1]:
                raise RuntimeError(f"Non-square patch map: {tuple(scores.shape)}")
            output["similarity_scores"] = scores.reshape(
                tokens.shape[0], 1, side, side
            )
        return output

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def context_from_model(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = "There is Pneumothorax",
) -> tuple[nn.Module, Tensor]:
    """Cache the one fixed official prompt using the frozen text encoder.

    The caller remains responsible for freezing the vision encoder and loading
    its checkpoint. This helper disables dropout and does not retain a text
    encoder graph or an inference-mode tensor.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    model.eval()
    loss_module = model.loss_fns["KeyPhraseAlignmentLoss"]
    device = next(model.parameters()).device
    encoded = tokenizer(
        [prompt], padding=True, truncation=True, return_tensors="pt"
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in ("input_ids", "attention_mask")
    }
    with torch.inference_mode(False), torch.no_grad():
        features, _ = loss_module.compute_text_features(
            [encoded], model.forward_text_model, ddp_gather=False
        )
        features = features.detach().clone()
    return loss_module, features


def build_head(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = "There is Pneumothorax",
    bottleneck: int = 32,
    scale: float = 0.1,
) -> RadZeroAdapterHead:
    """Build a head; the original encoders should be frozen by the caller."""
    loss_module, features = context_from_model(model, tokenizer, prompt)
    return RadZeroAdapterHead(loss_module, features, bottleneck=bottleneck, scale=scale)
