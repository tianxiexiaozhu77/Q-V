"""CheXchoNet-only unseen-query extension of the frozen MedKLIP adapter.

The three disease descriptions are fixed before evaluation. They are not
published author prompts, and no existing disease output substitutes for them.
The original 75 queries and every released parameter remain unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import torch

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[2]
BASE_PATH = HERE / 'adapter.py'
BASE_SHA256 = '3ed99e28f2882f6d8c3497d9fb0a503872204a7da3df73a0ad19ce7884cd7337'
PROMPTS_PATH = MEDICAL / 'AlphaRAD/local_setup/chexchonet_20261023/prompts.json'
PROMPTS_SHA256 = '9205cb15846cb39de94136909f6b0ab8c446a47cd4b4510ccb9c67bb0c99abf7'
for path, expected in ((BASE_PATH, BASE_SHA256), (PROMPTS_PATH, PROMPTS_SHA256)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Frozen CheXchoNet protocol input changed: {path}')
_spec = importlib.util.spec_from_file_location('_medklip_chexcho_frozen_base', BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
PROMPTS = json.loads(PROMPTS_PATH.read_text())
TASKS = list(PROMPTS['tasks'].values())
EXTRA = {task['class_name'].lower(): task['prompt'] for task in TASKS}


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Adapter(_base.Adapter):
    def __init__(self, device='cuda'):
        super().__init__(device)
        state_before = _state_digest(self.model)
        self.extra_book = dict(EXTRA)
        bert = str(_base.ROOT / 'local_assets/Bio_ClinicalBERT')
        with torch.inference_mode():
            tokens = self.tokenizer(list(EXTRA.values()), padding='max_length',
                                    truncation=True, max_length=64,
                                    return_tensors='pt').to(self.device)
            encoder = self.model._get_bert_basemodel(bert, freeze_layers=None)
            encoder = encoder.to(self.device).eval().requires_grad_(False)
            extra_queries = encoder(input_ids=tokens['input_ids'],
                                    attention_mask=tokens['attention_mask']).last_hidden_state[:, 0, :]
            self.model.disease_book = torch.cat([self.base_queries, extra_queries], 0)
        state_after = _state_digest(self.model)
        if state_after != state_before or not torch.equal(self.model.disease_book[:75], self.base_queries):
            raise RuntimeError('Appending CheXchoNet queries changed original weights or query context')
        self.classes = self.base_classes + list(EXTRA)
        self.active_mode = 'base75_plus_chexchonet3_fixed'
        self.metadata.update(
            protocol_revision='chexchonet_fixed_unseen_queries_v1',
            prompts='Original 75 descriptions plus three fixed CheXchoNet statements; Bio_ClinicalBERT CLS, max_length=64.',
            query_extension={
                'descriptions': EXTRA,
                'source': str(PROMPTS_PATH),
                'source_sha256': PROMPTS_SHA256,
                'exact_author_prompts': False,
                'semantic_label_source': 'https://physionet.org/content/chexchonet/1.0.0/',
                'query_count': len(self.classes),
                'protocol': 'Append explicitly named unseen diseases using the author text encoder and shared classifier; original 75-query context retained.',
                'not_used': 'No cardiomegaly/enlarge proxy, no fitted mapping, no test-label prompt optimization.',
            },
            base_adapter=str(BASE_PATH), base_adapter_sha256=BASE_SHA256,
            extension_integrity={'original_75_queries_bitwise_equal': True,
                                 'all_checkpoint_state_values_unchanged': True,
                                 'state_sha256_before': state_before,
                                 'state_sha256_after': state_after},
        )

    def check_classes(self, class_names):
        if not class_names or any(not isinstance(name, str) or name.lower() not in EXTRA
                                  for name in class_names):
            raise ValueError('This overlay accepts only the three fixed CheXchoNet disease names')
        return True

    def _indices(self, names):
        self.check_classes(names)
        if self.active_mode != 'base75_plus_chexchonet3_fixed':
            raise RuntimeError('CheXchoNet query context changed')
        return [self.classes.index(name.lower()) for name in names]


def create_adapter(device='cuda'):
    return Adapter(device)
