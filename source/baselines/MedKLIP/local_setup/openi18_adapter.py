"""Independent, frozen MedKLIP extension for all 18 OpenI labels.

The original 75 description embeddings and every checkpoint tensor are kept.
Two fixed descriptions are appended and use the existing shared projection,
decoder and binary classifier. Query self-attention can change old predictions.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEDICAL = HERE.parents[2]
RESOURCE_PATH = MEDICAL / "experiments/openi_query_extension/resources/openi_queries_v1.json"
QUERY_ORDER = ("calcified granuloma", "granuloma")
BASE_COUNT = 75
QUERY_COUNT = BASE_COUNT + len(QUERY_ORDER)
MODE = "base75_plus_openi2_fixed"


def _load_token_base():
    """Reuse the established collision-safe native-module loader."""
    name = "medklip_token_base_for_openi18"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "token_mlp_adapter.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


TOKEN_BASE = _load_token_base()
NATIVE_BASE = TOKEN_BASE._native_module()
ALIASES = dict(NATIVE_BASE.ALIASES, atelectases="atelectasis")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_digest(model):
    """Hash names, shapes, dtypes and exact tensor bytes, including buffers."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def read_resource():
    raw = RESOURCE_PATH.read_bytes()
    resource = json.loads(raw)
    if "version" not in resource or resource.get("query_order") != list(QUERY_ORDER):
        raise ValueError("OpenI resource must declare its version and the fixed two-query order")
    if set(resource.get("queries", {})) != set(QUERY_ORDER):
        raise ValueError("OpenI extension must contain exactly the two requested disease queries")
    for key in QUERY_ORDER:
        item = resource["queries"][key]
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ValueError(f"Missing OpenI disease description: {key}")
        if len(item.get("concepts", [])) != 8 or any(
                not isinstance(x, str) or not x.strip() for x in item["concepts"]):
            raise ValueError(f"Shared OpenI resource requires eight explicit concepts: {key}")
        if any(not item.get(field) for field in ("description_origin", "concept_policy", "sources")):
            raise ValueError(f"Missing OpenI query provenance: {key}")
    return resource, hashlib.sha256(raw).hexdigest()


def class_coverage(class_names):
    resource, resource_sha = read_resource()
    book = json.loads((ROOT / "PreTrain_MedKLIP/data_file/observation explanation.json").read_text())
    if len(book) != BASE_COUNT or set(book).intersection(QUERY_ORDER):
        raise RuntimeError("The expected original 75-query dictionary changed")
    if not class_names or any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError("At least one nonempty class name is required")
    keys = [ALIASES.get(name.lower(), name.lower()) for name in class_names]
    missing = [name for name, key in zip(class_names, keys)
               if key not in book and key not in resource["queries"]]
    return {
        "class_names": list(class_names), "query_keys": keys, "missing": missing,
        "full_task_supported": not missing,
        "additional_aliases": {"Atelectases": "atelectasis"},
        "query_policy": "Original 75 descriptions plus two separately versioned fixed OpenI descriptions",
        "resource": str(RESOURCE_PATH), "resource_sha256": resource_sha,
        "resource_version": resource["version"], "query_context_count": QUERY_COUNT,
        "appended_query_indices": dict(zip(QUERY_ORDER, range(BASE_COUNT, QUERY_COUNT))),
    }


class Adapter(NATIVE_BASE.Adapter):
    def __init__(self, device="cuda"):
        resource, resource_sha = read_resource()
        super().__init__(device)
        if len(self.base_classes) != BASE_COUNT or tuple(self.base_queries.shape) != (BASE_COUNT, 768):
            raise RuntimeError("Unexpected original MedKLIP query context")
        if set(self.base_classes).intersection(QUERY_ORDER):
            raise RuntimeError("An extension query already exists in the original dictionary")
        state_before = _state_digest(self.model)
        parameters_before = sum(p.numel() for p in self.model.parameters())
        prefix_before = hashlib.sha256(_tensor_bytes(self.base_queries)).hexdigest()
        self.extra_book = {key: resource["queries"][key]["description"] for key in QUERY_ORDER}
        descriptions = list(self.extra_book.values())
        untruncated = self.tokenizer(descriptions, padding=False, truncation=False,
                                     add_special_tokens=True)["input_ids"]
        tokens = self.tokenizer(descriptions, padding="max_length", truncation=True,
                                max_length=64, return_tensors="pt").to(self.device)
        lengths = {
            key: {"untruncated_length_including_special_tokens": len(ids),
                  "encoded_nonpadding_length": int(tokens["attention_mask"][i].sum().item()),
                  "max_length": 64, "truncated": len(ids) > 64}
            for i, (key, ids) in enumerate(zip(QUERY_ORDER, untruncated))
        }
        if lengths["granuloma"]["truncated"]:
            raise ValueError("The fixed Granuloma description must fit the native 64-token limit")
        bert_dir = ROOT / "local_assets/Bio_ClinicalBERT"
        # Ordinary no_grad tensors can participate in later adapter backprop.
        with torch.no_grad():
            encoder = self.model._get_bert_basemodel(str(bert_dir), freeze_layers=None)
            encoder = encoder.to(self.device).eval().requires_grad_(False)
            extra_queries = encoder(input_ids=tokens["input_ids"],
                                    attention_mask=tokens["attention_mask"]).last_hidden_state[:, 0, :]
            extra_queries = extra_queries.to(dtype=self.base_queries.dtype).detach().clone()
            if tuple(extra_queries.shape) != (2, 768) or not torch.isfinite(extra_queries).all():
                raise RuntimeError("Invalid appended disease embeddings")
            if torch.equal(extra_queries[0], extra_queries[1]):
                raise RuntimeError("The two disease descriptions produced identical query embeddings")
            self.extended_queries = torch.cat([self.base_queries, extra_queries], dim=0).detach().clone()
            self.model.disease_book = self.extended_queries
        del encoder
        state_after = _state_digest(self.model)
        parameters_after = sum(p.numel() for p in self.model.parameters())
        prefix_after = hashlib.sha256(_tensor_bytes(self.model.disease_book[:BASE_COUNT])).hexdigest()
        if state_before != state_after or parameters_before != parameters_after:
            raise RuntimeError("Appending queries changed released model parameters or buffers")
        if prefix_before != prefix_after or not torch.equal(self.model.disease_book[:BASE_COUNT], self.base_queries):
            raise RuntimeError("Appending queries changed the original 75 query embeddings")
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("The extended classifier must remain completely frozen")
        self.classes = self.base_classes + list(QUERY_ORDER)
        self.active_mode = MODE
        self.resource = deepcopy(resource)
        self.resource_sha256 = resource_sha
        source_files = [
            Path(__file__).resolve(), HERE / "adapter.py", HERE / "token_mlp_adapter.py",
            TOKEN_BASE.AUTHOR / "models/model_MedKLIP.py",
            TOKEN_BASE.AUTHOR / "models/transformer.py",
            TOKEN_BASE.AUTHOR / "models/tokenization_bert.py",
            ROOT / "PreTrain_MedKLIP/data_file/observation explanation.json",
            bert_dir / "config.json", bert_dir / "vocab.txt", RESOURCE_PATH,
        ]
        self.metadata.update(
            protocol_revision="medklip_openi18_fixed_query_extension_v1",
            prompts="Original 75 descriptions plus fixed Calcified Granuloma and Granuloma descriptions; Bio_ClinicalBERT CLS, max_length=64.",
            label_mapping=dict(ALIASES), query_context_count=QUERY_COUNT,
            original_query_count=BASE_COUNT, appended_query_count=2,
            original_queries_preserved=True,
            resource_sha256={str(RESOURCE_PATH): resource_sha},
            query_extension={
                "resource": str(RESOURCE_PATH), "source": str(RESOURCE_PATH),
                "resource_sha256": resource_sha, "source_sha256": resource_sha,
                "resource_version": resource["version"], "query_order": list(QUERY_ORDER),
                "query_indices": dict(zip(QUERY_ORDER, range(BASE_COUNT, QUERY_COUNT))),
                "queries": deepcopy(resource["queries"]), "descriptions": dict(self.extra_book),
                "text_tokenization": lengths, "query_count": QUERY_COUNT,
                "concepts_used_by_medklip": False,
                "protocol": "Append two description-derived queries; retain the original 75 and every released parameter; use the existing shared projection, decoder and classifier.",
                "old_prediction_invariance": False,
                "old_prediction_note": "Query self-attention couples diseases: unchanged original embeddings do not imply unchanged original-class predictions.",
                "new_disease_specific_parameters": 0,
            },
            extension_integrity={
                "original_75_queries_bitwise_equal": True,
                "all_checkpoint_state_values_unchanged": True,
                "state_sha256_before": state_before, "state_sha256_after": state_after,
                "query_prefix_sha256_before": prefix_before, "query_prefix_sha256_after": prefix_after,
                "parameters_before": parameters_before, "parameters_after": parameters_after,
                "gradient_trainable_parameters": 0,
            },
            source_files=[str(path) for path in source_files],
            source_sha256={str(path): _sha256(path) for path in source_files},
            checkpoint_files=[str(self.checkpoint), str(bert_dir / "pytorch_model.bin")],
            limitations=[
                "The two OpenI descriptions extend the original 75-query evaluation protocol; their provenance is recorded separately.",
                "Adding queries can change old-class predictions through query self-attention; the 77-query baseline must be evaluated separately.",
                "Fibrosis uses author CXR14 mapping tail_abnorm_obs, a broad abnormality concept.",
            ],
        )
        self._check_context()

    def _check_context(self):
        if (self.active_mode != MODE or self.model.disease_book is not self.extended_queries
                or tuple(self.model.disease_book.shape) != (QUERY_COUNT, 768)
                or self.classes != self.base_classes + list(QUERY_ORDER)):
            raise RuntimeError("The fixed 77-query OpenI context changed")

    def _activate_queries(self, covid=False):
        if covid:
            raise ValueError("This independent OpenI extension does not append a COVID query")
        self._check_context()

    def check_classes(self, class_names):
        if not class_names or any(not isinstance(name, str) or not name for name in class_names):
            raise ValueError("At least one nonempty class name is required")
        keys = [ALIASES.get(name.lower(), name.lower()) for name in class_names]
        missing = [name for name, key in zip(class_names, keys) if key not in self.classes]
        if missing:
            raise NotImplementedError("MedKLIP OpenI extension lacks: " + ", ".join(missing))
        if len(set(keys)) != len(keys):
            raise ValueError("Requested classes map to duplicate queries")
        return True

    def _indices(self, names):
        self._check_context()
        self.check_classes(names)
        return [self.classes.index(ALIASES.get(name.lower(), name.lower())) for name in names]


def create_adapter(device="cuda"):
    return Adapter(device)
