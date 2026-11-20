"""Fixed CheXchoNet text-query extension; no claimed author visual concepts.

The author did not release these three diseases' eight-concept resources. This
explicit experimental protocol repeats each fixed disease statement in all
eight concept slots and the description slot, retaining the nine-slot head,
the original 75 diseases' context, and every released checkpoint parameter.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import torch

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[2]
BASE_PATH = HERE / 'adapter.py'
BASE_SHA256 = 'd99770b1cd3a5a74345a0b5d746c55d6b59a3365637cd43a2c860c02fc81afe7'
PROMPTS_PATH = MEDICAL / 'AlphaRAD/local_setup/chexchonet_20261023/prompts.json'
PROMPTS_SHA256 = '9205cb15846cb39de94136909f6b0ab8c446a47cd4b4510ccb9c67bb0c99abf7'
for path, expected in ((BASE_PATH, BASE_SHA256), (PROMPTS_PATH, PROMPTS_SHA256)):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Frozen CheXchoNet protocol input changed: {path}')
_spec = importlib.util.spec_from_file_location('_mavl_chexcho_frozen_base', BASE_PATH)
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
        self.book.update(EXTRA)
        self.concepts.update({key: [text] * 8 for key, text in EXTRA.items()})
        self.metadata.update(
            protocol_revision='chexchonet_repeated_description_nine_slots_v1',
            prompts='Original 75 author disease descriptions/concepts plus three fixed CheXchoNet descriptions repeated in their nine native slots.',
            query_extension={
                'descriptions': EXTRA, 'source': str(PROMPTS_PATH),
                'source_sha256': PROMPTS_SHA256,
                'exact_author_prompts': False,
                'author_chexchonet_visual_concepts_available': False,
                'concept_policy': 'For each new disease, reuse its fixed description in all eight concept slots plus the main description slot. No invented visual findings and no test-label wording selection.',
                'semantic_label_source': 'https://physionet.org/content/chexchonet/1.0.0/',
                'checkpoint': 'checkpoint_full_46.pth; original feature-mode shared nine-slot head',
                'not_used': 'No cardiomegaly/enlarge proxy, no learned mapping, no prompt optimization.',
                'limitation': 'Experimental text-query extension rather than a strict replication of an author-specified CheXchoNet protocol; the original richer multi-aspect descriptions are unavailable for these new diseases.',
            },
            base_adapter=str(BASE_PATH), base_adapter_sha256=BASE_SHA256,
        )

    @torch.inference_mode()
    def _setup(self, mode):
        if mode != 'feature':
            raise NotImplementedError('This CheXchoNet overlay defines classification in original feature mode only')
        if self.active_mode == mode:
            return
        super()._setup(mode)
        if self.model.n_disease != 75 or self.model.n_concepts != 9:
            raise RuntimeError('Expected original 75 by 9 MAVL query layout')
        state_before = _state_digest(self.model)
        original_disease = self.model.disease_book
        original_concept = self.model.concept_book
        bert = str(_base.ROOT / 'local_assets/Bio_ClinicalBERT')
        tokenizer = _base.BertTokenizer.from_pretrained(bert, local_files_only=True)
        tokens = tokenizer(list(EXTRA.values()), padding='max_length', truncation=True,
                           max_length=64, return_tensors='pt').to(self.device)
        encoder = self.model._get_bert_basemodel(bert, freeze_layers=None)
        encoder = encoder.to(self.device).eval().requires_grad_(False)
        extra = encoder(input_ids=tokens['input_ids'],
                        attention_mask=tokens['attention_mask']).last_hidden_state[:, 0, :]
        self.model.disease_book = torch.cat([original_disease, extra], 0)
        self.model.concept_book = torch.cat([original_concept, extra.repeat_interleave(9, 0)], 0)
        self.model.n_disease = 78
        self.classes = self.base_classes + list(EXTRA)
        state_after = _state_digest(self.model)
        if state_after != state_before:
            raise RuntimeError('Appending CheXchoNet queries changed checkpoint state')
        if not torch.equal(self.model.disease_book[:75], original_disease):
            raise RuntimeError('Original MAVL disease queries changed')
        if not torch.equal(self.model.concept_book[:75 * 9], original_concept):
            raise RuntimeError('Original MAVL concept queries changed')
        self.metadata['extension_integrity'] = {
            'original_75_disease_queries_bitwise_equal': True,
            'original_675_concept_queries_bitwise_equal': True,
            'all_checkpoint_state_values_unchanged': True,
            'state_sha256_before': state_before, 'state_sha256_after': state_after,
            'query_count': 78, 'concepts_per_disease': 9,
        }

    def check_classes(self, class_names):
        if not class_names or any(not isinstance(name, str) or name.lower() not in EXTRA
                                  for name in class_names):
            raise ValueError('This overlay accepts only the three fixed CheXchoNet disease names')
        return True


def create_adapter(device='cuda'):
    return Adapter(device)
