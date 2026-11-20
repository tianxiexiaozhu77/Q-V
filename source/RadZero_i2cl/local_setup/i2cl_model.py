"""I2CL-style activation injection for the released, frozen RadZero model.

This is a vision adaptation, not a reproduction of the language-model I2CL
experiments.  The language model's last answer token is replaced by the visual
CLS token when averaging support-example activations.  The same context vector
is then broadcast to every token at each attention/MLP injection site:

    output = beta * output + alpha * context

Only the two scalar coefficients at each site are trainable.  The released
checkpoint has two alignment layers and two sites per layer, hence eight
trainable scalars.  The DINO encoder, text encoder, alignment weights, and
RadZero similarity scoring stay frozen.  DINO tokens before the alignment
transformer can be cached, but alignment outputs cannot be cached for training.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import math
from typing import Any

import torch
from torch import Tensor, nn


class RadZeroI2CL(nn.Module):
    """Keep RadZero scoring and learn scalar activation-injection coefficients.

    ``build_context`` must receive exactly the support examples subsequently
    used for coefficient calibration.  Dataset selection and supervision live
    in the caller; this class never reads evaluation labels or chooses samples.

    ``adaptation_state`` exports only the small adaptation state.  The ordinary
    PyTorch ``state_dict`` also includes the registered frozen model and should
    not be used to save a lightweight adaptation checkpoint.
    """

    MODULE_NAMES = ("attention", "mlp")
    STATE_VERSION = 1

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        prompts: Sequence[str],
        context_pooling: str = "cls",
        noise_scale: float = 0.001,
    ) -> None:
        super().__init__()
        if context_pooling != "cls":
            raise ValueError("This vision adaptation supports context_pooling='cls' only")
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if isinstance(prompts, str) or not prompts:
            raise ValueError("prompts must be a nonempty sequence of strings")
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("Each prompt must be a nonempty string")
        self.model = model.requires_grad_(False).eval()
        self.prompts = tuple(prompts)
        self.context_pooling = context_pooling
        self.noise_scale = float(noise_scale)
        self.class_count = len(self.prompts)
        self.context_sample_count = 0
        self.context_ready = False
        self._hooks_active = False

        alignment = self.model.align_transformer
        encoder = getattr(alignment, "transformer_layers", None)
        if encoder is None or not hasattr(encoder, "layer"):
            raise ValueError("RadZero requires an alignment transformer with layer blocks")
        self.layer_count = len(encoder.layer)
        if self.layer_count != 2:
            raise ValueError(
                f"This released-checkpoint protocol expects 2 alignment layers, got {self.layer_count}"
            )
        loss = self.model.loss_fns["KeyPhraseAlignmentLoss"]
        self.hidden_dim = int(loss.hidden_dim)
        for layer in encoder.layer:
            for name in self.MODULE_NAMES:
                if not isinstance(getattr(layer, name, None), nn.Module):
                    raise ValueError(f"Alignment layer is missing the {name!r} module")

        parameter = next(self.model.parameters())
        encoded = tokenizer(
            list(self.prompts), padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {
            name: values.to(parameter.device)
            for name, values in encoded.items()
            if name in ("input_ids", "attention_mask")
        }
        # Match the existing RadZero batch inference path and its fixed prompts.
        split_prompts = [
            {name: values[index:index + 1] for name, values in encoded.items()}
            for index in range(self.class_count)
        ]
        # Normal, detached tensors remain usable by autograd even when a caller
        # constructed the wrapper from inside an outer inference_mode context.
        with torch.inference_mode(False), torch.no_grad():
            features, _ = loss.compute_text_features(
                split_prompts, self.model.forward_text_model, ddp_gather=False
            )
            features = features.detach().clone()
            if features.shape != (self.class_count, self.hidden_dim):
                raise ValueError(f"Unexpected text-feature shape: {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise ValueError("Non-finite text features")
            self.register_buffer("text_features", features)
            self.register_buffer(
                "context_vectors",
                torch.zeros(
                    self.layer_count, len(self.MODULE_NAMES), self.hidden_dim,
                    device=features.device, dtype=features.dtype,
                ),
            )
            coefficients = torch.empty(
                self.layer_count, len(self.MODULE_NAMES), 2,
                device=features.device, dtype=features.dtype,
            )
            coefficients[..., 0] = 0.1  # alpha, the context coefficient
            coefficients[..., 1] = 1.0  # beta, the original activation coefficient
            self.coefficients = nn.Parameter(coefficients)

    @property
    def device(self) -> torch.device:
        return self.coefficients.device

    @property
    def dtype(self) -> torch.dtype:
        return self.coefficients.dtype

    def train(self, mode: bool = True) -> RadZeroI2CL:
        super().train(mode)
        # Frozen modules must not acquire dropout or stochastic-depth updates.
        self.model.eval()
        return self

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _validate_tokens(self, tokens: Tensor) -> None:
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] == 0
            or tokens.shape[1] < 2
            or tokens.shape[2] != self.hidden_dim
            or not tokens.is_floating_point()
        ):
            shape = tuple(tokens.shape) if isinstance(tokens, Tensor) else type(tokens)
            raise ValueError(
                f"Expected nonempty floating [batch, tokens, {self.hidden_dim}], got {shape}"
            )

    def _sites(self) -> Iterator[tuple[int, int, nn.Module]]:
        layers = self.model.align_transformer.transformer_layers.layer
        for layer_index, layer in enumerate(layers):
            for module_index, name in enumerate(self.MODULE_NAMES):
                yield layer_index, module_index, getattr(layer, name)

    @contextmanager
    def _temporary_hooks(self, hook_factory: Any) -> Iterator[None]:
        if self._hooks_active:
            raise RuntimeError("Nested or concurrent I2CL hook contexts are unsupported")
        handles = []
        self._hooks_active = True
        try:
            for layer_index, module_index, module in self._sites():
                handles.append(
                    module.register_forward_hook(hook_factory(layer_index, module_index))
                )
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._hooks_active = False

    @staticmethod
    def _activation(outputs: Any) -> Tensor:
        activation = outputs[0] if isinstance(outputs, tuple) else outputs
        if not isinstance(activation, Tensor) or activation.ndim != 3:
            raise RuntimeError("Expected an alignment module output of shape [B,T,D]")
        return activation

    def encode_images(self, pixels: Tensor) -> Tensor:
        """Return frozen DINO tokens *before* the alignment transformer.

        At the released 518-pixel resolution the shape is [B,1370,768].
        Tokens returned here can be reused during all calibration epochs.
        """
        self.model.eval()
        with torch.inference_mode(False), torch.no_grad():
            outputs = self.model.vision_model(pixels.to(device=self.device, dtype=self.dtype))
            tokens = outputs["last_hidden_state"].detach()
            self._validate_tokens(tokens)
            if not torch.isfinite(tokens).all():
                raise RuntimeError("Non-finite frozen DINO tokens")
        return tokens

    def build_context(self, tokens: Tensor, batch_size: int = 2) -> Tensor:
        """Average each unmodified module's CLS activation over support images.

        The input must contain every support image exactly once, in the same
        support set used to calibrate coefficients.  Context collection never
        injects coefficients, uses labels, or adds training noise.
        """
        self._validate_tokens(tokens)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        count = len(tokens)
        self.model.eval()
        with torch.inference_mode(False), torch.no_grad():
            sums = torch.zeros(
                self.layer_count, len(self.MODULE_NAMES), self.hidden_dim,
                device=self.device, dtype=torch.float64,
            )
            seen = [[0] * len(self.MODULE_NAMES) for _ in range(self.layer_count)]

            def capture(layer_index: int, module_index: int) -> Any:
                def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
                    del module, inputs
                    activation = self._activation(outputs)
                    if activation.shape[-1] != self.hidden_dim:
                        raise RuntimeError("Unexpected alignment activation hidden dimension")
                    cls = activation[:, 0, :].detach()
                    if not torch.isfinite(cls).all():
                        raise RuntimeError("Non-finite support activation")
                    sums[layer_index, module_index].add_(cls.double().sum(dim=0))
                    seen[layer_index][module_index] += len(cls)
                return hook

            with self._temporary_hooks(capture):
                for first in range(0, count, batch_size):
                    batch = tokens[first:first + batch_size].to(
                        device=self.device, dtype=self.dtype
                    )
                    if not torch.isfinite(batch).all():
                        raise ValueError("Non-finite support tokens")
                    self.model.align_transformer(batch)
            if any(value != count for layer in seen for value in layer):
                raise RuntimeError(f"Support hook coverage mismatch: {seen}, expected {count}")
            context = (sums / count).to(dtype=self.dtype)
            if not torch.isfinite(context).all():
                raise RuntimeError("Non-finite averaged context vectors")
            self.context_vectors.copy_(context)
        self.context_sample_count = count
        self.context_ready = True
        return self.context_vectors

    def _injection_hook(self, layer_index: int, module_index: int, noise: bool) -> Any:
        def hook(module: nn.Module, inputs: Any, outputs: Any) -> Any:
            del module, inputs
            activation = self._activation(outputs)
            alpha, beta = self.coefficients[layer_index, module_index].unbind()
            context = self.context_vectors[layer_index, module_index].view(1, 1, -1)
            # Preserve autograd through the frozen stack to earlier coefficients.
            # A detached activation here would silently break that dependency.
            injected = beta * activation + alpha * context
            if noise and self.noise_scale:
                # Match I2CL's norm-scaled noise, with no derivative through its
                # scale. Noise is disabled for context extraction and evaluation.
                norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True).detach()
                injected = injected + torch.randn_like(injected) * norm * self.noise_scale
            if isinstance(outputs, tuple):
                return (injected,) + outputs[1:]
            return injected
        return hook

    def forward_tokens(
        self,
        tokens: Tensor,
        inject: bool = True,
        return_maps: bool = False,
        training_noise: bool = False,
    ) -> dict[str, Tensor]:
        """Score pre-alignment DINO tokens without detaching the alignment graph.

        Raw similarity maps use the same ``similarity_scores`` key as RadZero.
        Set ``inject=False`` for the original frozen baseline.  Training noise
        requires both ``training_noise=True`` and ``self.training=True``.
        """
        self._validate_tokens(tokens)
        if tokens.device != self.device or tokens.dtype != self.dtype:
            raise ValueError(
                f"Tokens must use {self.device}/{self.dtype}, got {tokens.device}/{tokens.dtype}"
            )
        self.model.eval()
        if inject:
            if not self.context_ready:
                raise RuntimeError("Call build_context or load_adaptation_state before injection")
            if not torch.isfinite(self.coefficients).all():
                raise RuntimeError("Non-finite injection coefficients")
            noise = bool(training_noise and self.training)
            with self._temporary_hooks(
                lambda layer, module: self._injection_hook(layer, module, noise)
            ):
                vision = self.model.align_transformer(tokens)
        else:
            if self._hooks_active:
                raise RuntimeError("A baseline forward cannot run while injection hooks are active")
            vision = self.model.align_transformer(tokens)

        loss = self.model.loss_fns["KeyPhraseAlignmentLoss"]
        if loss.layer_norm is not None:
            vision = loss.layer_norm(vision)
        if not loss.use_vision_cls_token:
            vision = vision[:, 1:]
        logits, attention = loss.compute_t2i_logits(
            self.text_features, vision, need_attn_weights=return_maps
        )
        logits = logits.reshape(self.class_count, len(tokens)).T / loss.loss_temperature.exp()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite I2CL logits")
        result = {"logits": logits}
        if return_maps:
            if attention is None or len(attention) == 0:
                raise RuntimeError("Released RadZero scoring returned no similarity maps")
            scores = torch.stack(attention).mean(dim=0)
            if loss.use_vision_cls_token:
                scores = scores[:, :, 1:]
            side = math.isqrt(scores.shape[-1])
            if side * side != scores.shape[-1]:
                raise RuntimeError(f"Non-square patch map: {tuple(scores.shape)}")
            maps = scores.reshape(len(tokens), self.class_count, side, side)
            if not torch.isfinite(maps).all():
                raise RuntimeError("Non-finite I2CL similarity maps")
            result["similarity_scores"] = maps
        return result

    def forward(
        self,
        pixels: Tensor,
        inject: bool = True,
        return_maps: bool = False,
        training_noise: bool = False,
    ) -> dict[str, Tensor]:
        return self.forward_tokens(
            self.encode_images(pixels),
            inject=inject,
            return_maps=return_maps,
            training_noise=training_noise,
        )

    @contextmanager
    def identity_coefficients(self) -> Iterator[None]:
        """Temporarily use alpha=0/beta=1 for reference equivalence checks."""
        saved = self.coefficients.detach().clone()
        with torch.no_grad():
            self.coefficients[..., 0].zero_()
            self.coefficients[..., 1].fill_(1)
        try:
            yield
        finally:
            with torch.no_grad():
                self.coefficients.copy_(saved)

    def adaptation_state(self) -> dict[str, Any]:
        """Export context and coefficients, without any released model weights."""
        if not self.context_ready or self.context_sample_count <= 0:
            raise RuntimeError("Cannot save adaptation state before collecting context")
        return {
            "state_version": self.STATE_VERSION,
            "method": "radzero-align-i2cl-vision-cls",
            "module_names": list(self.MODULE_NAMES),
            "layer_count": self.layer_count,
            "hidden_dim": self.hidden_dim,
            "context_pooling": self.context_pooling,
            "context_sample_count": self.context_sample_count,
            "noise_scale": self.noise_scale,
            "prompts": list(self.prompts),
            "coefficients": self.coefficients.detach().cpu().clone(),
            "context_vectors": self.context_vectors.detach().cpu().clone(),
        }

    def load_adaptation_state(self, state: Mapping[str, Any]) -> None:
        """Load a small checkpoint, validating architecture and prompt order."""
        expected = {
            "state_version": self.STATE_VERSION,
            "method": "radzero-align-i2cl-vision-cls",
            "module_names": list(self.MODULE_NAMES),
            "layer_count": self.layer_count,
            "hidden_dim": self.hidden_dim,
            "context_pooling": self.context_pooling,
            "prompts": list(self.prompts),
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Adaptation checkpoint {key!r} mismatch")
        count = state.get("context_sample_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Invalid context sample count in adaptation checkpoint")
        noise_scale = state.get("noise_scale")
        if (
            not isinstance(noise_scale, (int, float))
            or not math.isfinite(noise_scale)
            or noise_scale < 0
        ):
            raise ValueError("Invalid noise scale in adaptation checkpoint")
        tensors = {}
        for name in ("coefficients", "context_vectors"):
            value = state.get(name)
            reference = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != reference.shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid {name!r} in adaptation checkpoint")
            tensors[name] = value.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self.coefficients.copy_(tensors["coefficients"])
            self.context_vectors.copy_(tensors["context_vectors"])
        self.context_sample_count = count
        self.noise_scale = float(noise_scale)
        self.context_ready = True
