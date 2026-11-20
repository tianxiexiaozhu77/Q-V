"""Frozen 77-query MedKLIP backend for the independent OpenI18 experiment."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent


def _extension_module():
    name = "medklip_openi18_native_for_token_mlp"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "openi18_adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


EXTENSION = _extension_module()
class_coverage = EXTENSION.class_coverage


class TokenBackend(EXTENSION.TOKEN_BASE.TokenBackend):
    def __init__(self, device="cuda"):
        # The inherited scorer is query-count agnostic. Construct only one model.
        self.native = EXTENSION.Adapter(device)
        self.device = self.native.device
        self.model = self.native.model
        self.class_names = None
        self.image_encoder_released = False
        self.metadata = deepcopy(self.native.metadata)
        self.metadata.update({
            "native_score_kind": "positive_class_softmax_probability",
            "score_kind": "binary_logit_margin",
            "score_transform": "logit_positive - logit_negative; sigmoid(margin) equals native positive softmax",
            "insertion": "After native image_encoder/res_l2; before the frozen original disease-query Transformer decoder",
            "token_shape": [196, self.dim], "native_query_context": EXTENSION.QUERY_COUNT,
            "original_query_count": EXTENSION.BASE_COUNT, "query_context_count": EXTENSION.QUERY_COUNT,
            "dim": self.dim, "aggregation_origin": "native",
            "cache_batch_size": self.recommended_cache_batch_size,
            "train_microbatch_size": self.recommended_train_batch_size,
            "aggregation": "Native frozen MedKLIP four-layer decoder with 75 original and two fixed appended queries; not FLaS",
            "frozen": "Visual encoder/projection, all 77 text queries/projection, decoder and classifier",
            "trainable": "Only the caller-supplied shared residual token adapter",
        })
        path = str(Path(__file__).resolve())
        self.metadata["source_files"].append(path)
        self.metadata["source_sha256"][path] = EXTENSION._sha256(path)

    def prepare_classes(self, class_names):
        audit = class_coverage(class_names)
        if audit["resource_sha256"] != self.native.resource_sha256:
            raise RuntimeError("The OpenI query resource changed after model construction")
        if audit["missing"]:
            raise NotImplementedError("MedKLIP OpenI extension lacks: " + ", ".join(audit["missing"]))
        self.indices = self.native._indices(class_names)
        self.class_names = tuple(class_names)
        self.native_names = tuple(class_names)
        # Do not use the old prepare_classes: it asserts a 75-query context.
        with torch.no_grad():
            self.queries = self.model.disease_embedding_layer(self.model.disease_book).detach().clone()
        if tuple(self.queries.shape) != (EXTENSION.QUERY_COUNT, self.dim):
            raise RuntimeError("Unexpected projected 77-query shape")
        if not torch.isfinite(self.queries).all():
            raise RuntimeError("The projected disease queries contain nonfinite values")
        self.metadata.update(class_coverage=audit, class_query_indices=list(self.indices),
                             class_names=list(class_names), query_context_count=EXTENSION.QUERY_COUNT)
        return self

    def _check_prepared(self):
        super()._check_prepared()
        self.native._check_context()

    @torch.no_grad()
    def native_predictions(self, image_paths):
        self._check_prepared()
        # The independent native wrapper preserves the 77-query context on every call.
        return torch.as_tensor(self.native.predict(image_paths, self.native_names), device=self.device)
