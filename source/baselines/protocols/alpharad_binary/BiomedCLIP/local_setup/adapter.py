"""Frozen pretrained model with disease-wise presence-versus-absence scoring.

This independent protocol revision retains native image/text encoders and transforms.
It does not replace the archived positive-only diagnostic implementation.
"""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINES_ROOT = Path(__file__).resolve().parents[4]
MODEL_NAME = 'BiomedCLIP'
BASE_ADAPTER_SHA = '42e4d2176650561f37715edf71d77b9dcfd628e9a1daaa5649a2a626ae123470'
BASE_MODEL_ROOT = BASELINES_ROOT / MODEL_NAME
BASE_ADAPTER_PATH = BASE_MODEL_ROOT / 'local_setup/adapter.py'
if hashlib.sha256(BASE_ADAPTER_PATH.read_bytes()).hexdigest() != BASE_ADAPTER_SHA:
    raise RuntimeError('The referenced frozen base adapter changed; audit this protocol before proceeding')
_spec = importlib.util.spec_from_file_location('_frozen_base_' + MODEL_NAME.replace('-', '_'), BASE_ADAPTER_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def disease_phrases(class_names):
    key = tuple(class_names)
    if not key or any(not isinstance(name, str) or not name.strip() for name in key):
        raise ValueError('Nonempty disease names required')
    return tuple(name.replace('_', ' ') for name in key)


class Adapter(_base.Adapter):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self._binary_text = {}
        self.metadata.update(
            model=MODEL_NAME,
            protocol_revision='alpharad_binary_v1',
            score_kind='per_disease_negative_positive_raw_cosine_binary_softmax_probability',
            positive_prompt='There is {disease}.',
            negative_prompt='There is no {disease}.',
            prompt_template='There is {disease}.',
            prompt_templates=['There is no {disease}.', 'There is {disease}.'],
            softmax_axis='two hypotheses independently for each disease',
            temperature=1.0,
            logit_scale_used=False,
            class_mapping='replace underscores with spaces only; output names/order unchanged',
            protocol='AlphaRAD binary helper formula applied to native frozen model cosine features. Positive/negative presence comparison follows BioViL author classification example. Native image preprocessing retained; not claimed to exactly reproduce undocumented baseline details in the AlphaRAD table.',
            revision_reason='Positive-only cosine diagnostic omitted explicit absence comparison required by the chosen disease classification protocol.',
            base_adapter=str(BASE_ADAPTER_PATH),
            base_adapter_sha256=BASE_ADAPTER_SHA,
            protocol_sources=['AlphaRAD/cxrclip/base_evaluator.py:434-464', 'BioViL-T/official/hi-ml-multimodal/test_multimodal/vlp/test_zero_shot_classification.py:147-157'],
        )

    @torch.inference_mode()
    def predict(self, image_paths, class_names):
        phrases = disease_phrases(class_names)
        if phrases not in self._binary_text:
            prompts = [f'There is no {name}.' for name in phrases] + [f'There is {name}.' for name in phrases]
            tokens = self.tokenizer(prompts).to(self.device)
            self._binary_text[phrases] = self.model.encode_text(tokens, normalize=True)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(self.preprocess(image.convert('RGB')))
        features = self.model.encode_image(torch.stack(images).to(self.device), normalize=True)
        cosine = features @ self._binary_text[phrases].T
        count = len(phrases)
        binary = torch.stack([cosine[:, :count], cosine[:, count:]], dim=-1)
        return binary.softmax(dim=-1)[..., 1].float().cpu().numpy()


def create_adapter(device='cuda'):
    return Adapter(device)
