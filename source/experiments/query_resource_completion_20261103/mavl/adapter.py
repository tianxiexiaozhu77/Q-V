"""Append only frozen missing disease descriptions to native MAVL context."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT / 'baselines/MAVL/local_setup/adapter.py'
NATIVE_SHA = 'd99770b1cd3a5a74345a0b5d746c55d6b59a3365637cd43a2c860c02fc81afe7'
DESCRIPTIONS = HERE.parent / 'query_descriptions.json'
DESCRIPTIONS_SHA = 'aa7e8ce55477f6b8aee09f11448fdb08688b0a89a0ef9e2652e43c55aa4fd778'
for path, expected in ((NATIVE, NATIVE_SHA), (DESCRIPTIONS, DESCRIPTIONS_SHA)):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
spec = importlib.util.spec_from_file_location('_mavl_resource_frozen_base', NATIVE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
FROZEN = json.loads(DESCRIPTIONS.read_text())


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Adapter(base.Adapter):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.extra = {}
        self.original_book = dict(self.book)
        self.original_concepts = dict(self.concepts)
        self.extension_arrays = {}
        self._native_transform = self.transform
        self._batch_paths = None
        self._batch_images = {}
        self._batch_tensors = {}
        self.transform = self._cached_transform
        self.metadata.update(
            protocol_revision=FROZEN['version'],
            prompts='Original 75 disease descriptions/concepts plus only the missing target descriptions repeated in nine slots.',
            limitations=[x for x in self.metadata['limitations'] if 'TB tasks rejected' not in x],
            frozen_descriptions=str(DESCRIPTIONS), frozen_descriptions_sha256=DESCRIPTIONS_SHA,
            extension_policy='Fixed study-defined descriptions, not author-released concepts. Preserve all 75 original diseases and nine slots per disease. No PadChest rare-class resources.',
        )

    def configure(self, class_names):
        extra = {}
        for name in class_names:
            key = base.ALIASES.get(name.lower(), name.lower())
            if key not in self.base_classes:
                if name not in FROZEN['descriptions']:
                    raise ValueError('Missing fixed description for ' + name)
                extra[key] = FROZEN['descriptions'][name]
        if extra != self.extra:
            self.active_mode = None
            self.extra = extra
            self.book = {**self.original_book, **extra}
            self.concepts = {**self.original_concepts, **{k: [v] * 8 for k,v in extra.items()}}
            self.extension_arrays = {}
        self.metadata['new_descriptions'] = dict(extra)
        self.metadata['new_class_order'] = list(extra)
        self.check_classes(class_names)

    def cache_batch(self, paths):
        """Cache only this batch's exact native decoded/preprocessed tensors."""
        ordered = tuple(map(str, paths))
        if ordered == self._batch_paths:
            return
        self._batch_paths = ordered
        self._batch_images = {path: super(Adapter, self)._read_image(path) for path in ordered}
        self._batch_tensors = {id(image): self._native_transform(image)
                               for image in self._batch_images.values()}

    def _read_image(self, path):
        if str(path) in self._batch_images:
            return self._batch_images[str(path)]
        return super()._read_image(path)

    def _cached_transform(self, image):
        value = self._batch_tensors.get(id(image))
        return value if value is not None else self._native_transform(image)

    def verify_pixel_cache(self, paths):
        self.cache_batch(paths)
        for path in paths:
            expected = self._native_transform(super()._read_image(str(path)))
            actual = self._cached_transform(self._read_image(str(path)))
            assert torch.equal(actual, expected), 'Native preprocessing cache changed pixels'
        return dict(passed=True, images=len(paths), bitwise_equal=True, max_abs_error=0.)

    @torch.inference_mode()
    def _setup(self, mode):
        if mode != 'feature':
            raise NotImplementedError('This overlay supports native classification feature mode only')
        if self.active_mode == mode:
            return
        super()._setup(mode)
        assert self.model.n_disease == 75 and self.model.n_concepts == 9
        before = state_digest(self.model)
        disease, concept = self.model.disease_book, self.model.concept_book
        self.extension_arrays = dict(original_disease_book=disease.cpu().numpy().copy(),
                                     original_concept_book=concept.cpu().numpy().copy())
        if self.extra:
            bert = str(base.ROOT / 'local_assets/Bio_ClinicalBERT')
            tokenizer = base.BertTokenizer.from_pretrained(bert, local_files_only=True)
            tokens = tokenizer(list(self.extra.values()), padding='max_length', truncation=True,
                               max_length=64, return_tensors='pt').to(self.device)
            encoder = self.model._get_bert_basemodel(bert, freeze_layers=None)
            encoder = encoder.to(self.device).eval().requires_grad_(False)
            embeddings = encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask']).last_hidden_state[:, 0, :]
            self.model.disease_book = torch.cat([disease, embeddings], 0)
            self.model.concept_book = torch.cat([concept, embeddings.repeat_interleave(9, 0)], 0)
            self.model.n_disease = 75 + len(self.extra)
            self.classes = self.base_classes + list(self.extra)
            self.extension_arrays['new_description_embeddings'] = embeddings.cpu().numpy().copy()
            self.extension_arrays['token_ids'] = tokens['input_ids'].cpu().numpy().copy()
            self.extension_arrays['attention_mask'] = tokens['attention_mask'].cpu().numpy().copy()
            del encoder
        assert torch.equal(self.model.disease_book[:75], disease)
        assert torch.equal(self.model.concept_book[:675], concept)
        after = state_digest(self.model)
        assert before == after
        self.extension_arrays.update(disease_book=self.model.disease_book.cpu().numpy().copy(),
                                     concept_book=self.model.concept_book.cpu().numpy().copy())
        self.metadata['extension_integrity'] = dict(
            original_75_disease_queries_bitwise_equal=True,
            original_675_concept_queries_bitwise_equal=True,
            all_checkpoint_state_values_unchanged=True,
            state_sha256_before=before, state_sha256_after=after,
            query_count=self.model.n_disease, concepts_per_disease=9,
            new_query_count=len(self.extra), new_class_order=list(self.extra),
            context_caveat='Appending query context can affect existing disease scores through native self-attention; all four branches share the same context.',
        )


def create_adapter(device='cuda'):
    return Adapter(device)
