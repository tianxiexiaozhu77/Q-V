"""Fixed OpenI18 query extension of the released frozen MAVL classifier.

Calcified Granuloma uses the author's description and eight visual concepts.
Granuloma uses one fixed NLM-derived description in all nine slots.  This
explicitly degenerate second query is not an author eight-concept resource.
The shared classifier and all original checkpoint tensors stay unchanged.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
MEDICAL = ROOT.parent.parent
AUTHOR = ROOT / "Zero-shot_classification"
BASE_PATH = ROOT / "local_setup/adapter.py"
BASE_SHA256 = "d99770b1cd3a5a74345a0b5d746c55d6b59a3365637cd43a2c860c02fc81afe7"
RESOURCE_PATH = MEDICAL / "experiments/openi_query_extension/resources/openi_queries_v1.json"
RESOURCE_SHA256 = "b14b3833e2b621dff709cb411cb62fa9fb51dd465ef4f77db5faf9370f921c40"
QUERY_ORDER = ("calcified granuloma", "granuloma")
SLOT_NAMES = ("description", "border", "fluid", "location", "opacity", "other", "patterns", "shape", "texture")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_module():
    if file_sha256(BASE_PATH) != BASE_SHA256:
        raise RuntimeError("Frozen MAVL base adapter changed")
    name = "mavl_native_for_openi18_extension"
    if name in sys.modules:
        return sys.modules[name]
    saved = {key: value for key, value in sys.modules.items()
             if key == "models" or key.startswith("models.")}
    paths = list(sys.path)
    for key in saved:
        del sys.modules[key]
    try:
        spec = importlib.util.spec_from_file_location(name, BASE_PATH)
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


_base = _base_module()
ALIASES = dict(_base.ALIASES, atelectases="atelectasis")


def load_resource():
    if file_sha256(RESOURCE_PATH) != RESOURCE_SHA256:
        raise RuntimeError("Frozen OpenI18 query resource changed")
    resource = json.loads(RESOURCE_PATH.read_text())
    if tuple(resource["query_order"]) != QUERY_ORDER or set(resource["queries"]) != set(QUERY_ORDER):
        raise ValueError("OpenI18 extension requires the two fixed queries in their frozen order")
    for key in QUERY_ORDER:
        query = resource["queries"][key]
        if not isinstance(query["description"], str) or not query["description"].strip():
            raise ValueError("Missing disease description: " + key)
        if len(query["concepts"]) != 8 or any(not isinstance(t, str) or not t.strip() for t in query["concepts"]):
            raise ValueError("Each disease requires eight nonempty concept-slot texts: " + key)
        for source in query["sources"]:
            if "path" in source and file_sha256(source["path"]) != source["sha256"]:
                raise RuntimeError("Frozen query source changed: " + source["path"])
    calcified = resource["queries"][QUERY_ORDER[0]]
    author_book = json.loads((AUTHOR / "observation explanation_padchest_rare.json").read_text())
    author_concepts = json.loads((ROOT / "Pretrain/concept_gen/gpt4_mimic_padchest_rare.json").read_text())
    if calcified["description"] != author_book[QUERY_ORDER[0]] or calcified["concepts"] != author_concepts[QUERY_ORDER[0]]:
        raise ValueError("Calcified Granuloma must retain the exact author description and ordered concepts")
    granuloma = resource["queries"][QUERY_ORDER[1]]
    if granuloma["concepts"] != [granuloma["description"]] * 8:
        raise ValueError("This protocol requires Granuloma's eight repeated-description slots")
    return resource


def token_length_audit(tokenizer, resource):
    """Record lengths including special tokens before the native 64-token cap."""
    records = {}
    for key in QUERY_ORDER:
        query = resource["queries"][key]
        texts = [query["description"], *query["concepts"]]
        ids = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)["input_ids"]
        lengths = [len(sequence) for sequence in ids]
        if key == "granuloma" and max(lengths) > 64:
            raise ValueError("The fixed Granuloma description must fit in 64 tokens including special tokens")
        records[key] = {"slot_order": list(SLOT_NAMES), "untruncated_lengths": lengths,
                        "includes_special_tokens": True, "max_length": 64,
                        "truncated_slots": [slot for slot, length in zip(SLOT_NAMES, lengths) if length > 64]}
    return records


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Adapter(_base.Adapter):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.resource = load_resource()
        self.book.update({key: self.resource["queries"][key]["description"] for key in QUERY_ORDER})
        self.concepts.update({key: self.resource["queries"][key]["concepts"] for key in QUERY_ORDER})
        source_files = [Path(__file__).resolve(), BASE_PATH,
                        AUTHOR / "models/model_MAVL.py", AUTHOR / "models/transformer.py",
                        AUTHOR / "models/clip.py", AUTHOR / "models/tokenization_bert.py",
                        AUTHOR / "observation explanation.json",
                        ROOT / "Pretrain/concept_gen/gpt4_mimic.json",
                        AUTHOR / "observation explanation_padchest_rare.json",
                        ROOT / "Pretrain/concept_gen/gpt4_mimic_padchest_rare.json", RESOURCE_PATH,
                        ROOT / "local_setup/python_local.sh"]
        bert = ROOT / "local_assets/Bio_ClinicalBERT"
        resources = [RESOURCE_PATH, *[p for p in source_files if p.suffix == ".json"],
                     bert / "config.json", bert / "vocab.txt", bert / "pytorch_model.bin",
                     ROOT / "local_assets/checkpoint_full_46.pth"]
        self.metadata.update({
            "protocol_revision": self.resource["version"],
            "score_kind": "positive_class_softmax_probability from the fixed OpenI18 query extension",
            "label_mapping": ALIASES,
            "prompts": "Original 75 author disease descriptions and 675 concept queries; append author Calcified Granuloma nine slots and fixed NLM-derived Granuloma description repeated in nine slots.",
            "query_context_count": 693, "original_query_count": 75,
            "original_concept_query_count": 675, "extended_disease_count": 77,
            "concepts_per_disease": 9, "original_queries_preserved": False,
            "source_files": [str(p) for p in source_files],
            "source_sha256": {str(p): file_sha256(p) for p in source_files},
            "resource_sha256": {str(p): file_sha256(p) for p in dict.fromkeys(resources)},
            "checkpoint_files": [str(ROOT / "local_assets/checkpoint_full_46.pth")],
            "query_extension": {
                "resource": str(RESOURCE_PATH), "resource_sha256": RESOURCE_SHA256,
                "query_order": list(QUERY_ORDER), "queries": deepcopy(self.resource["queries"]),
                "mixed_resource_protocol": True, "exact_author_prompts_for_both_classes": False,
                "granuloma_protocol": "Nine repeated-description slots: one new fixed NLM-derived description plus eight identical copies. This is a degenerate text-query extension, not an author eight-concept scheme.",
                "checkpoint": "Original checkpoint_full_46.pth, feature mode, unchanged shared 2304-to-2 head",
                "context_effect": "Decoder self-attention includes all 693 queries; adding queries can change original-class scores even though their original embeddings and checkpoint weights are unchanged.",
                "baseline_label": "MAVL + fixed OpenI18 query extension (without Token MLP)",
                "fixed_before_new_test_evaluation": self.resource.get("fixed_before_new_test_evaluation", False),
            },
            "limitations": [*self.metadata["limitations"], *self.resource.get("limitations", [])],
        })

    @torch.no_grad()
    def _setup(self, mode):
        if mode != "feature":
            raise NotImplementedError("The fixed OpenI18 extension supports only feature classification")
        if self.active_mode == mode:
            if self.model.n_disease != 77 or self.model.n_concepts != 9:
                raise RuntimeError("OpenI18 feature context was unexpectedly replaced")
            return
        super()._setup(mode)
        if self.model.n_disease != 75 or self.model.n_concepts != 9:
            raise RuntimeError("Expected the original 75-by-9 MAVL query layout")
        state_before = _state_digest(self.model)
        original_disease = self.model.disease_book.detach().clone()
        original_concepts = self.model.concept_book.detach().clone()
        original_projected = self.model.disease_embedding_layer(original_concepts).detach().clone()
        bert = str(ROOT / "local_assets/Bio_ClinicalBERT")
        tokenizer = _base.BertTokenizer.from_pretrained(bert, local_files_only=True)
        lengths = token_length_audit(tokenizer, self.resource)
        texts = [text for key in QUERY_ORDER for text in
                 [self.resource["queries"][key]["description"], *self.resource["queries"][key]["concepts"]]]
        tokens = tokenizer(texts, padding="max_length", truncation=True,
                           max_length=64, return_tensors="pt").to(self.device)
        encoder = self.model._get_bert_basemodel(bert, freeze_layers=None)
        encoder = encoder.to(self.device).eval().requires_grad_(False)
        extra = encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state[:, 0, :].detach()
        if extra.shape != (18, 768):
            raise RuntimeError("Unexpected additional MAVL concept embedding shape")
        self.model.disease_book = torch.cat([original_disease, extra[[0, 9]]], dim=0)
        self.model.concept_book = torch.cat([original_concepts, extra], dim=0)
        self.model.n_disease = 77
        self.classes = self.base_classes + list(QUERY_ORDER)
        state_after = _state_digest(self.model)
        if state_after != state_before:
            raise RuntimeError("Appending OpenI18 queries changed checkpoint state")
        if not torch.equal(self.model.disease_book[:75], original_disease):
            raise RuntimeError("Original MAVL disease embeddings changed")
        if not torch.equal(self.model.concept_book[:675], original_concepts):
            raise RuntimeError("Original MAVL concept embeddings changed")
        if not torch.equal(self.model.disease_embedding_layer(self.model.concept_book[:675]), original_projected):
            raise RuntimeError("Original projected MAVL queries changed")
        if tuple(self.model.classifier.weight.shape) != (2, 2304):
            raise RuntimeError("The original nine-slot classifier shape changed")
        self.metadata.update(original_queries_preserved=True, untruncated_token_lengths=lengths,
                             extension_integrity={
                                 "original_75_disease_queries_bitwise_equal": True,
                                 "original_675_concept_queries_bitwise_equal": True,
                                 "original_675_projected_queries_bitwise_equal": True,
                                 "all_checkpoint_state_values_unchanged": True,
                                 "state_sha256_before": state_before, "state_sha256_after": state_after,
                                 "original_disease_book_sha256": _tensor_digest(original_disease),
                                 "original_concept_book_sha256": _tensor_digest(original_concepts),
                                 "extended_concept_book_sha256": _tensor_digest(self.model.concept_book),
                                 "query_count": 77, "concept_query_count": 693,
                                 "concepts_per_disease": 9, "classifier_shape": [2, 2304],
                             })

    def _keys(self, class_names):
        if not class_names or any(not isinstance(name, str) for name in class_names):
            raise ValueError("At least one disease name is required")
        keys = [ALIASES.get(name.lower(), name.lower()) for name in class_names]
        missing = [key for key in keys if key not in self.base_classes and key not in QUERY_ORDER]
        if missing:
            raise NotImplementedError("MAVL OpenI18 feature query unavailable: " + ", ".join(missing))
        if len(keys) != len(set(keys)):
            raise ValueError("Requested diseases map to duplicate queries")
        return keys

    def check_classes(self, class_names):
        self._keys(class_names)
        return True

    @torch.no_grad()
    def predict(self, image_paths, class_names):
        keys = self._keys(class_names)
        self._setup("feature")
        indices = [self.classes.index(key) for key in keys]
        pixels = torch.stack([self.transform(self._read_image(str(path))) for path in image_paths]).to(self.device)
        logits = self.model(pixels, return_intermediate=False)
        return logits.softmax(dim=-1)[:, indices, 1].cpu().numpy()


def create_adapter(device="cuda"):
    return Adapter(device)
