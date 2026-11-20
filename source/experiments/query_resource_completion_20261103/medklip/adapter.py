"""Append only frozen study-defined missing MedKLIP queries; no weight changes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL_ROOT = ROOT / 'baselines/MedKLIP'
BASE_PATH = MODEL_ROOT / 'local_setup/adapter.py'
BASE_SHA = '3ed99e28f2882f6d8c3497d9fb0a503872204a7da3df73a0ad19ce7884cd7337'
DESCRIPTIONS_PATH = HERE.parent / 'query_descriptions.json'
DESCRIPTIONS_SHA = 'aa7e8ce55477f6b8aee09f11448fdb08688b0a89a0ef9e2652e43c55aa4fd778'
for path, expected in ((BASE_PATH, BASE_SHA), (DESCRIPTIONS_PATH, DESCRIPTIONS_SHA)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen source changed: ' + str(path))
spec = importlib.util.spec_from_file_location('_medklip_frozen_resource_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
DESCRIPTIONS = json.loads(DESCRIPTIONS_PATH.read_text())['descriptions']


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
        assert len(self.base_classes) == len(self.base_queries) == 75
        self.initial_state_sha256 = state_digest(self.model)
        self.initial_base_queries = self.base_queries.detach().clone()
        self.frozen_extra = {}
        self.context_history = []
        self.metadata.update(
            protocol_revision='fixed_missing_query_resources_20261103_v1',
            prompts='Original75 author descriptions plus only missing dataset classes from the fixed study-defined description JSON; Bio_ClinicalBERT CLS max_length64.',
            query_extension={'path': str(DESCRIPTIONS_PATH), 'sha256': DESCRIPTIONS_SHA,
                'study_defined': True, 'author_released_prompts': False,
                'selection_performed': False, 'all_four_branches_share_context': True},
            limitations=['Fibrosis retains author CXR14 tail_abnorm_obs mapping.',
                'Added descriptions are study-defined resources. Query self-attention may change existing disease scores when context is extended.'])

    def check_classes(self, names):
        missing = [name for name in names if base.ALIASES.get(name.lower(), name.lower()) not in self.book
                   and name not in DESCRIPTIONS]
        if missing:
            raise ValueError('No frozen query description: ' + ', '.join(missing))
        return True

    @torch.inference_mode()
    def _indices(self, names):
        self.check_classes(names)
        keys = [base.ALIASES.get(name.lower(), name.lower()) for name in names]
        needed = [(name, key) for name, key in zip(names, keys) if key not in self.book]
        if len({key for _, key in needed}) != len(needed):
            raise ValueError('Duplicate requested missing query')
        mode = 'base75' if not needed else 'base75_plus_frozen:' + '|'.join(key for _, key in needed)
        if self.active_mode != mode:
            absent = [(name, key) for name, key in needed if key not in self.frozen_extra]
            if absent:
                tokens = self.tokenizer([DESCRIPTIONS[name] for name, _ in absent],
                    padding='max_length', truncation=True, max_length=64, return_tensors='pt').to(self.device)
                encoder = self.model._get_bert_basemodel(str(MODEL_ROOT / 'local_assets/Bio_ClinicalBERT'), None)
                encoder = encoder.to(self.device).eval().requires_grad_(False)
                encoded = encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask']).last_hidden_state[:, 0, :]
                for (_, key), vector in zip(absent, encoded):
                    self.frozen_extra[key] = vector.detach().clone()
                del encoder
            self.model.disease_book = (torch.cat([self.base_queries, torch.stack([self.frozen_extra[key] for _, key in needed])])
                                       if needed else self.base_queries)
            self.classes = self.base_classes + [key for _, key in needed]
            self.active_mode = mode
            self.context_history.append({'mode': mode, 'classes_added': [name for name, _ in needed],
                'descriptions': [DESCRIPTIONS[name] for name, _ in needed], 'query_count': len(self.classes)})
        if not torch.equal(self.model.disease_book[:75], self.initial_base_queries):
            raise RuntimeError('Original75 query vectors changed')
        return [self.classes.index(key) for key in keys]

    def integrity(self):
        current = state_digest(self.model)
        return {'original_75_queries_bitwise_equal': bool(torch.equal(self.model.disease_book[:75], self.initial_base_queries)),
                'state_sha256_before': self.initial_state_sha256, 'state_sha256_after': current,
                'all_checkpoint_state_values_unchanged': current == self.initial_state_sha256,
                'active_mode': self.active_mode, 'query_context_count': len(self.classes),
                'context_history': self.context_history}


def create_adapter(device='cuda'):
    return Adapter(device)
