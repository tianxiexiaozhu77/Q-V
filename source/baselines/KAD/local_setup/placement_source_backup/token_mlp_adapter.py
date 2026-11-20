"""Projected patch-token adapter boundary before frozen KAD's native TQN.

This keeps the original text queries, attention layers and two-class readout.
The training logit is z_positive-z_negative, whose sigmoid exactly represents
the released positive-class softmax probability. No new FLaS module is used.
"""
from pathlib import Path
import importlib.util

import torch
from PIL import Image


def _load_native():
    path = Path(__file__).with_name("adapter.py")
    spec = importlib.util.spec_from_file_location("kad_token_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter


class TokenBackend:
    dim = 768

    def __init__(self, device="cuda"):
        self.native = _load_native()(device=device)
        self.device = self.native.device
        self.model = self.native.model
        self.class_names = None
        self.metadata = dict(self.native.metadata)
        self.metadata.update({
            "adapter_insertion": "image encoder projected 768-dimensional patch tokens, before native TQN LayerNorm",
            "token_boundary": "ModelRes projected spatial tokens; before TQN_Model.decoder_norm",
            "aggregation_origin": "native",
            "dim": self.dim,
            "token_layout": "B x 196 x 768 projected ResNet spatial tokens; no CLS token",
            "aggregation": "native text-query TQN cross-attention; not AlphaRAD FLaS",
            "adapter_sharing": "one shared residual MLP for all image tokens and diseases",
            "native_score_kind": "positive_class_softmax_probability",
            "score_kind": "positive logit minus negative logit",
            "training_score_kind": "logit",
            "score_transformation": "sigmoid(z_positive-z_negative) equals native softmax(z)[positive]",
            "score_transform": "positive-minus-negative binary logit; sigmoid recovers native probability",
            "frozen": "complete author image/text encoders, projection, TQN and classifier; eval mode",
            "python_launcher": str(Path(__file__).with_name("python_local.sh")),
            "source_files": [str(Path(__file__)), str(Path(__file__).with_name("adapter.py")),
                             str(Path(__file__).parents[1] / "A3_CLIP/models/clip_tqn.py")],
            "cache_batch": 32,
            "training_batch": 16,
            "cache_batch_size": 32,
            "train_microbatch_size": 8,
        })

    @torch.no_grad()
    def prepare_classes(self, class_names):
        self.native.check_classes(class_names)
        self.class_names = tuple(class_names)
        # no_grad, rather than inference_mode, creates ordinary detached
        # tensors that can be saved by autograd through the frozen decoder.
        self.query = self.native._queries(class_names).detach().clone()
        self.metadata["class_names"] = list(class_names)
        aliases = self.native.metadata["label_mapping"]
        self.metadata["query_texts"] = [aliases.get(name.lower(), name.lower()) for name in class_names]

    @torch.no_grad()
    def encode_images(self, image_paths):
        if not image_paths:
            raise ValueError("Expected at least one image")
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(self.native.transform(image.convert("RGB")))
        features, _ = self.native.image_encoder(torch.stack(images).to(self.device))
        return features.detach()

    def score_tokens(self, tokens, token_adapter=None):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before scoring")
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(f"Expected B x N x {self.dim} tokens, got {tuple(tokens.shape)}")
        adapted = tokens if token_adapter is None else token_adapter(tokens)
        pair_logits = self.model(adapted, self.query)
        return pair_logits[..., 1] - pair_logits[..., 0]

    @torch.no_grad()
    def native_scores(self, image_paths):
        # Recompute the untouched original image/TQN path independently of
        # score_tokens, returning the equivalent binary training logit.
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before scoring")
        features = self.encode_images(image_paths)
        pair_logits = self.native.model(features, self.native._queries(self.class_names))
        return pair_logits[..., 1] - pair_logits[..., 0]

    @torch.no_grad()
    def native_predictions(self, image_paths):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before scoring")
        result = self.native.predict(image_paths, self.class_names)
        return torch.as_tensor(result, device=self.device)

    @torch.no_grad()
    def native_scores_from_tokens(self, tokens):
        return self.score_tokens(tokens)

    def release_image_encoder(self):
        """Move encoders to CPU after caching; queries and TQN stay put."""
        self.native.image_encoder.cpu()
        self.native.text_encoder.cpu()


def create_backend(device="cuda"):
    return TokenBackend(device=device)
