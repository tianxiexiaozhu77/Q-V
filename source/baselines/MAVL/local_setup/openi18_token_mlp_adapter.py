"""Shared pre-decoder Token MLP on the fixed mixed-resource MAVL OpenI18 extension.

The original 675 queries plus 18 appended queries are used by both controls and
adapted scores.  No result is represented as the original author's OpenI18 model.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
TOKEN_BASE_SHA256 = "14efae2f49b2454a0e4e984b2d2621f618082efcc152d2255abd254a5f1b7fb6"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


_extension = _load("mavl_openi18_native_extension", HERE / "openi18_adapter.py")
if _extension.file_sha256(HERE / "token_mlp_adapter.py") != TOKEN_BASE_SHA256:
    raise RuntimeError("Frozen MAVL Token MLP backend changed")
_token_base = _load("mavl_token_backend_for_openi18", HERE / "token_mlp_adapter.py")


def class_coverage(class_names):
    resource = _extension.load_resource()
    book = json.loads((_extension.AUTHOR / "observation explanation.json").read_text())
    concepts = json.loads((_extension.ROOT / "Pretrain/concept_gen/gpt4_mimic.json").read_text())
    keys = [_extension.ALIASES.get(name.lower(), name.lower()) for name in class_names]
    available = ((set(book) & set(concepts)) - {"covid19"}) | set(resource["query_order"])
    missing = [name for name, key in zip(class_names, keys) if key not in available]
    return {"class_names": list(class_names), "query_keys": keys, "missing": missing,
            "full_task_supported": not missing,
            "additional_aliases": {"Atelectases": "atelectasis"},
            "query_policy": "Retain original 75-by-9 query prefix; append author Calcified Granuloma nine slots and fixed NLM-derived Granuloma description in nine repeated slots.",
            "protocol_revision": resource["version"],
            "resource_sha256": {str(_extension.RESOURCE_PATH): _extension.RESOURCE_SHA256},
            "query_context_count": 693, "original_query_count": 75}


class TokenBackend(_token_base.TokenBackend):
    def __init__(self, device="cuda"):
        self.native = _extension.Adapter(device)
        self.device = self.native.device
        self.class_names = None
        self.image_encoder_released = False
        self.metadata = deepcopy(self.native.metadata)
        self.metadata.update({
            "native_score_kind": "fixed_query_extension_positive_class_softmax_probability",
            "score_kind": "binary_logit_margin",
            "score_transform": "logit_positive - logit_negative; sigmoid(margin) equals fixed-extension positive softmax",
            "insertion": "After native image_encoder/res_l2; before frozen MAVL decoder with all 693 extended concept queries",
            "token_shape": [196, self.dim], "dim": self.dim,
            "native_query_context": 693, "original_native_query_context": 675,
            "query_context_count": 693,
            "original_query_count": 75, "aggregation_origin": "native",
            "cache_batch_size": self.recommended_cache_batch_size,
            "train_microbatch_size": self.recommended_train_batch_size,
            "aggregation": "Native frozen MAVL four-layer decoder with original 675 queries and 18 fixed extension queries; not FLaS",
            "frozen": "Visual encoder/projection, original and appended concept queries, projection, full decoder and nine-slot classifier",
            "trainable": "Only the caller-supplied shared residual Token MLP",
            "unused_branches": "Anatomical/global auxiliary outputs are omitted from feature-mode token scoring",
            "baseline_protocol": "MAVL + fixed OpenI18 query extension; same mixed-resource 693-query context as Token MLP",
        })
        for path in (Path(__file__).resolve(), HERE / "token_mlp_adapter.py"):
            self.metadata["source_files"].append(str(path))
            self.metadata["source_sha256"][str(path)] = _extension.file_sha256(path)

    def prepare_classes(self, class_names):
        if not class_names:
            raise ValueError("At least one class is required")
        audit = class_coverage(class_names)
        if audit["missing"]:
            raise NotImplementedError("MAVL OpenI18 fixed-extension query unavailable: " + ", ".join(audit["missing"]))
        self.native.check_classes(class_names)
        self.native._setup("feature")
        self.model = self.native.model
        self.indices = [self.native.classes.index(key) for key in audit["query_keys"]]
        self.class_names = tuple(class_names)
        self.native_names = tuple(class_names)
        with torch.no_grad():
            self.queries = self.model.disease_embedding_layer(self.model.concept_book).detach().clone()
        if self.model.n_disease != 77 or self.model.n_concepts != 9 or self.queries.shape != (693, self.dim):
            raise RuntimeError("Expected extended 77-by-9 MAVL concept context")
        if self.model.decoder_name != "cross":
            raise RuntimeError("Expected the original MAVL cross-attention decoder")
        for key in ("original_queries_preserved", "untruncated_token_lengths", "extension_integrity"):
            self.metadata[key] = deepcopy(self.native.metadata[key])
        self.metadata.update(class_coverage=audit, class_query_indices=self.indices,
                             class_names=list(class_names), query_context_count=693)
        return self

    @torch.no_grad()
    def native_predictions(self, image_paths):
        self._check_prepared()
        return torch.as_tensor(self.native.predict(image_paths, self.native_names), device=self.device)
