"""Trainable-token boundary for the frozen, released MedKLIP classifier.

The adapter is supplied by the caller.  Its output is the memory for all 75
native disease queries, not a replacement for the original query decoder.
Scores are binary logit margins, whose sigmoid equals native positive softmax.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "Sample_zero-shot_Classification_CXR14"


def _native_module():
    """Avoid collisions with other baselines' generically named models package."""
    name = "medklip_native_for_token_mlp"
    if name in sys.modules:
        return sys.modules[name]
    saved = {k: v for k, v in sys.modules.items() if k == "models" or k.startswith("models.")}
    paths = list(sys.path)
    for key in saved:
        del sys.modules[key]
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / "local_setup/adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
        return module
    finally:
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = paths


def class_coverage(class_names):
    native = _native_module()
    book = json.loads((ROOT / "PreTrain_MedKLIP/data_file/observation explanation.json").read_text())
    aliases = dict(native.ALIASES, atelectases="atelectasis")
    keys = [aliases.get(name.lower(), name.lower()) for name in class_names]
    missing = [name for name, key in zip(class_names, keys) if key not in book]
    return {"class_names": list(class_names), "query_keys": keys, "missing": missing,
            "full_task_supported": not missing,
            "additional_aliases": {"Atelectases": "atelectasis"},
            "query_policy": "Original 75 descriptions only; no invented or substituted missing queries."}


class TokenBackend:
    dim = 256
    recommended_cache_batch_size = 32
    recommended_train_batch_size = 32

    def __init__(self, device="cuda"):
        self.native = _native_module().Adapter(device)
        self.device = self.native.device
        self.model = self.native.model
        self.class_names = None
        self.image_encoder_released = False
        self.metadata = deepcopy(self.native.metadata)
        self.metadata.update({
            "native_score_kind": "positive_class_softmax_probability",
            "score_kind": "binary_logit_margin",
            "score_transform": "logit_positive - logit_negative; sigmoid(margin) equals native positive softmax",
            "insertion": "After native image_encoder/res_l2; before original disease-query Transformer decoder",
            "token_shape": [196, self.dim], "native_query_context": 75,
            "dim": self.dim, "aggregation_origin": "native",
            "cache_batch_size": self.recommended_cache_batch_size,
            "train_microbatch_size": self.recommended_train_batch_size,
            "aggregation": "Native frozen MedKLIP four-layer disease-query decoder; not FLaS",
            "frozen": "Visual encoder/projection, text queries/projection, full decoder and classifier",
            "trainable": "Only the caller-supplied shared residual token adapter",
            "source_files": [str(Path(__file__).resolve()), str(ROOT / "local_setup/adapter.py"),
                             str(AUTHOR / "models/model_MedKLIP.py"), str(AUTHOR / "models/transformer.py"),
                             str(ROOT / "PreTrain_MedKLIP/data_file/observation explanation.json")],
            "checkpoint_files": [str(self.native.checkpoint)],
        })

    def prepare_classes(self, class_names):
        audit = class_coverage(class_names)
        if audit["missing"]:
            raise NotImplementedError("MedKLIP native description resources missing: " + ", ".join(audit["missing"]))
        if not class_names:
            raise ValueError("At least one class is required")
        self.native._activate_queries(False)
        self.indices = [self.native.classes.index(key) for key in audit["query_keys"]]
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Requested classes map to duplicate native queries")
        self.class_names = tuple(class_names)
        self.native_names = tuple("Atelectasis" if name.lower() == "atelectases" else name for name in class_names)
        # no_grad, rather than inference_mode: the fixed queries will participate
        # in a differentiable forward through the frozen decoder during training.
        with torch.no_grad():
            self.queries = self.model.disease_embedding_layer(self.model.disease_book).detach().clone()
        assert self.queries.shape == (75, self.dim)
        self.metadata.update(class_coverage=audit, class_query_indices=self.indices,
                             class_names=list(class_names), query_context_count=75)
        return self

    def _check_prepared(self):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before encoding/scoring")

    def _pixels(self, paths):
        if self.image_encoder_released:
            raise RuntimeError("Image encoder was released; use cached tokens")
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.native.transform(image.convert("RGB")))
        return torch.stack(images).to(self.device)

    @torch.no_grad()
    def encode_images(self, image_paths):
        self._check_prepared()
        return self.model.image_encoder(self._pixels(image_paths)).detach()

    def score_tokens(self, tokens, token_adapter=None):
        self._check_prepared()
        if tokens.ndim != 3 or tokens.shape[1:] != (196, self.dim):
            raise ValueError(f"Expected [B,196,{self.dim}] projected tokens; received {tuple(tokens.shape)}")
        tokens = tokens.to(device=self.device, dtype=self.queries.dtype)
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        query = self.queries.unsqueeze(1).repeat(1, tokens.shape[0], 1)
        features, _ = self.model.decoder(query, tokens.transpose(0, 1),
                                         memory_key_padding_mask=None, pos=None, query_pos=None)
        logits = self.model.classifier(self.model.dropout_feas(features)).transpose(0, 1)
        return (logits[..., 1] - logits[..., 0])[:, self.indices]

    @torch.no_grad()
    def native_scores(self, image_paths):
        self._check_prepared()
        logits = self.model(self._pixels(image_paths))
        return (logits[..., 1] - logits[..., 0])[:, self.indices]

    @torch.no_grad()
    def native_predictions(self, image_paths):
        self._check_prepared()
        return torch.as_tensor(self.native.predict(image_paths, self.native_names), device=self.device)

    @torch.no_grad()
    def native_scores_from_tokens(self, tokens):
        return self.score_tokens(tokens)

    def release_image_encoder(self):
        self._check_prepared()
        for name in ("res_features", "res_l1", "res_l2", "resnet_dict"):
            if hasattr(self.model, name):
                delattr(self.model, name)
        self.image_encoder_released = True
