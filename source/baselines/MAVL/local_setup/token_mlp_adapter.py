"""Frozen MAVL classifier with an optional pre-decoder shared token adapter.

All 75 x 9 native concept queries remain in the decoder.  Only the ordinary
feature-classification path is supported; the COVID global/text path is not
silently reinterpreted as a token-based classifier.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "Zero-shot_classification"


def _native_module():
    name = "mavl_native_for_token_mlp"
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
    book = json.loads((AUTHOR / "observation explanation.json").read_text())
    concepts = json.loads((ROOT / "Pretrain/concept_gen/gpt4_mimic.json").read_text())
    aliases = dict(native.ALIASES, atelectases="atelectasis")
    keys = [aliases.get(name.lower(), name.lower()) for name in class_names]
    missing = [name for name, key in zip(class_names, keys)
               if key == "covid19" or key not in book or key not in concepts]
    return {"class_names": list(class_names), "query_keys": keys, "missing": missing,
            "full_task_supported": not missing,
            "additional_aliases": {"Atelectases": "atelectasis"},
            "query_policy": "Original 75 diseases and all nine concepts; no invented or substituted missing queries."}


class TokenBackend:
    dim = 256
    recommended_cache_batch_size = 32
    # The 675 native queries make decoder gradients heavier than MedKLIP.
    # Accumulate four microbatches to retain the common effective batch of 32.
    recommended_train_batch_size = 8

    def __init__(self, device="cuda"):
        self.native = _native_module().Adapter(device)
        self.device = self.native.device
        self.class_names = None
        self.image_encoder_released = False
        self.metadata = deepcopy(self.native.metadata)
        self.metadata.update({
            "native_score_kind": "positive_class_softmax_probability",
            "score_kind": "binary_logit_margin",
            "score_transform": "logit_positive - logit_negative; sigmoid(margin) equals native positive softmax",
            "insertion": "After native image_encoder/res_l2; before original disease/concept Transformer decoder",
            "token_shape": [196, self.dim], "native_query_context": 675,
            "dim": self.dim, "aggregation_origin": "native",
            "cache_batch_size": self.recommended_cache_batch_size,
            "train_microbatch_size": self.recommended_train_batch_size,
            "aggregation": "Native frozen MAVL four-layer decoder with all 75 x 9 concept queries; not FLaS",
            "frozen": "Visual encoder/projection, concept queries/projection, full decoder and classifier",
            "trainable": "Only the caller-supplied shared residual token adapter",
            "unused_branches": "Anatomical/global auxiliary outputs are omitted because feature-mode classification does not consume them",
            "source_files": [str(Path(__file__).resolve()), str(ROOT / "local_setup/adapter.py"),
                             str(AUTHOR / "models/model_MAVL.py"), str(AUTHOR / "models/transformer.py"),
                             str(AUTHOR / "observation explanation.json"),
                             str(ROOT / "Pretrain/concept_gen/gpt4_mimic.json")],
            "checkpoint_files": [str(ROOT / "local_assets/checkpoint_full_46.pth")],
        })

    def prepare_classes(self, class_names):
        audit = class_coverage(class_names)
        if audit["missing"]:
            raise NotImplementedError("MAVL native feature-mode description/concept resources missing: " + ", ".join(audit["missing"]))
        if not class_names:
            raise ValueError("At least one class is required")
        self.native._setup("feature")
        self.model = self.native.model
        self.indices = [self.native.classes.index(key) for key in audit["query_keys"]]
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Requested classes map to duplicate native queries")
        self.class_names = tuple(class_names)
        self.native_names = tuple("Atelectasis" if name.lower() == "atelectases" else name for name in class_names)
        with torch.no_grad():
            self.queries = self.model.disease_embedding_layer(self.model.concept_book).detach().clone()
        assert self.model.n_disease == 75 and self.model.n_concepts == 9
        assert self.queries.shape == (675, self.dim)
        assert self.model.decoder_name == "cross"
        self.metadata.update(class_coverage=audit, class_query_indices=self.indices,
                             class_names=list(class_names), query_context_count=675)
        return self

    def _check_prepared(self):
        if self.class_names is None:
            raise RuntimeError("Call prepare_classes before encoding/scoring")

    def _pixels(self, paths):
        if self.image_encoder_released:
            raise RuntimeError("Image encoder was released; use cached tokens")
        return torch.stack([self.native.transform(self.native._read_image(str(path))) for path in paths]).to(self.device)

    @torch.no_grad()
    def encode_images(self, image_paths):
        self._check_prepared()
        spatial, _ = self.model.image_encoder(self._pixels(image_paths))
        return spatial.permute(0, 2, 3, 1).reshape(spatial.shape[0], -1, self.dim).detach()

    def score_tokens(self, tokens, token_adapter=None):
        self._check_prepared()
        if tokens.ndim != 3 or tokens.shape[1:] != (196, self.dim):
            raise ValueError(f"Expected [B,196,{self.dim}] projected tokens; received {tuple(tokens.shape)}")
        tokens = tokens.to(device=self.device, dtype=self.queries.dtype)
        if token_adapter is not None:
            tokens = token_adapter(tokens)
        batch = tokens.shape[0]
        query = self.queries.unsqueeze(1).repeat(1, batch, 1)
        features, _ = self.model.decoder(query, tokens.transpose(0, 1))
        out = self.model.dropout_feas(features).permute(1, 2, 0)
        if self.model.bottneck:
            out = self.model.bottleneck(out)
        out = out.permute(0, 2, 1).contiguous().view(batch, self.model.n_disease, -1)
        logits = self.model.classifier(out)
        return (logits[..., 1] - logits[..., 0])[:, self.indices]

    @torch.no_grad()
    def native_scores(self, image_paths):
        self._check_prepared()
        logits = self.model(self._pixels(image_paths), return_intermediate=False)
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
        for name in ("backbone", "pool", "res_l1", "res_l2", "global_l"):
            if hasattr(self.model, name):
                delattr(self.model, name)
        self.image_encoder_released = True
