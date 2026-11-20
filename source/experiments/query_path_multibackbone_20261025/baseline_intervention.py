"""Native visual-update interventions for decoder/classifier CXR backbones.

These models have no independently query-conditioned final readout in the
classification interface used here. Their counterfactual is F (fixed visual
aggregation), NOT AlphaRAD/RadZero's readout-only R. F feeds q' to the native
model and replaces every visual cross-attention update with its original-q
update for the same image. Query residuals, self attention, FFNs, layer norms
and the pretrained classifier remain native. CARZero only clamps image-to-text
visual updates; its text-to-image direction is exactly the Q direction.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = ('CARZero', 'KAD', 'MedKLIP', 'MAVL')
SOURCES = {
    'CARZero': 'baselines/CARZero/CARZero/models/dqn_wo_self_atten_mlp.py',
    'KAD': 'baselines/KAD/A3_CLIP/models/clip_tqn.py',
    'MedKLIP': 'baselines/MedKLIP/Sample_zero-shot_Classification_CXR14/models/model_MedKLIP.py',
    'MAVL': 'baselines/MAVL/Zero-shot_classification/models/model_MAVL.py',
}
DECODER_SOURCES = {
    'CARZero': 'baselines/CARZero/CARZero/models/transformer_decoder.py',
    'KAD': 'baselines/KAD/A3_CLIP/models/transformer_decoder.py',
    'MedKLIP': 'baselines/MedKLIP/Sample_zero-shot_Classification_CXR14/models/transformer.py',
    'MAVL': 'baselines/MAVL/Zero-shot_classification/models/transformer.py',
}


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    if value is None:
        return None
    raise TypeError(f'Unsupported attention output type: {type(value)}')


def _check_equal(actual, expected, context):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise RuntimeError(f'{context}: invalid shape or nonfinite prediction')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.allclose(actual, expected, rtol=1e-5, atol=2e-6):
        raise RuntimeError(f'{context}: native numerical parity failed, max_abs_error={error}')
    return dict(passed=True, max_abs_error=error, rtol=1e-5, atol=2e-6)


class VisualUpdateTape:
    """Record/replay each native cross-attention output, including all layers.

    Hook scopes always remove handles on exceptions. Keys and values must match
    the original visual memory exactly on replay; only query-side state may
    differ. One call per registered layer is required by the audited decoders.
    """
    def __init__(self, decoder):
        if decoder.training:
            raise ValueError('Interventions require an evaluation-mode frozen decoder')
        self.modules = [(f'layers.{i}.multihead_attn', layer.multihead_attn)
                        for i, layer in enumerate(decoder.layers)]
        if not self.modules or len({id(m) for _, m in self.modules}) != len(self.modules):
            raise ValueError('Expected distinct visual cross-attention modules for all decoder layers')
        if any(not isinstance(m, torch.nn.MultiheadAttention) or m.training for _, m in self.modules):
            raise TypeError('Only native evaluation-mode torch MultiheadAttention is supported')
        self.records = []
        self.last_replay_count = 0

    @staticmethod
    def _memory(args, kwargs):
        key = kwargs.get('key', args[1] if len(args) > 1 else None)
        value = kwargs.get('value', args[2] if len(args) > 2 else None)
        if not torch.is_tensor(key) or not torch.is_tensor(value):
            raise RuntimeError('Native attention did not expose visual key/value tensors')
        return key, value

    @contextmanager
    def record(self):
        self.records = []
        handles = []
        def capture(index):
            def hook(module, args, kwargs, output):
                if index != len(self.records):
                    raise RuntimeError('Cross-attention execution order or multiplicity changed')
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError('Expected native attention update and weights tuple')
                key, value = self._memory(args, kwargs)
                self.records.append((_clone_tree(output), key.detach().clone(), value.detach().clone()))
            return hook
        try:
            for index, (_, module) in enumerate(self.modules):
                handles.append(module.register_forward_hook(capture(index), with_kwargs=True))
            yield self
            if len(self.records) != len(self.modules):
                raise RuntimeError('Not all native visual cross-attention layers were recorded')
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def replay(self):
        if len(self.records) != len(self.modules):
            raise RuntimeError('A complete original-query visual update tape is required')
        handles = []
        self.last_replay_count = 0
        def replace(index):
            def hook(module, args, kwargs, output):
                if index != self.last_replay_count:
                    raise RuntimeError('Cross-attention replay order or multiplicity changed')
                expected, original_key, original_value = self.records[index]
                key, value = self._memory(args, kwargs)
                if not torch.equal(key, original_key) or not torch.equal(value, original_value):
                    raise RuntimeError('Fixed aggregation attempted to replay on different visual memory')
                if output[0].shape != expected[0].shape:
                    raise RuntimeError('Fixed aggregation output shape differs from native attention')
                self.last_replay_count += 1
                return _clone_tree(expected)
            return hook
        try:
            for index, (_, module) in enumerate(self.modules):
                handles.append(module.register_forward_hook(replace(index), with_kwargs=True))
            yield self
            if self.last_replay_count != len(self.modules):
                raise RuntimeError('Not all native visual cross-attention layers were replayed')
        finally:
            for handle in handles:
                handle.remove()

    def audit(self):
        return dict(layer_count=len(self.modules), layer_names=[n for n, _ in self.modules],
                    recorded_layers=len(self.records), replayed_layers=self.last_replay_count,
                    visual_key_values_exactly_equal=True)


@contextmanager
def _shared_visual_encoding(native, model_name):
    """Reuse identical native pixels/features across Original, F and Q calls."""
    if model_name == 'CARZero':
        owner, attr = native.model, 'image_encoder_forward'
    elif model_name == 'KAD':
        owner, attr = native.image_encoder, 'forward'
    else:
        owner, attr = native.model, 'image_encoder'
    old = getattr(owner, attr)
    had_override = attr in owner.__dict__
    cached = []
    def once(pixels):
        if not cached:
            cached.append((pixels.detach().clone(), old(pixels)))
        elif not torch.equal(pixels, cached[0][0]):
            raise ValueError('Intervention branches received different preprocessed pixels')
        return cached[0][1]
    setattr(owner, attr, once)
    try:
        yield
    finally:
        if had_override:
            setattr(owner, attr, old)
        else:
            delattr(owner, attr)


class BaselineIntervention:
    def __init__(self, model_name, native, query_adapter):
        if model_name not in SUPPORTED:
            raise ValueError(f'Unsupported decoder intervention model: {model_name}')
        self.model_name, self.native, self.query = model_name, native, query_adapter
        source = ROOT / SOURCES[model_name]
        self.metadata = dict(
            model=model_name, control_name='fixed_aggregation', control_symbol='F',
            contrast='Q-F', readout_only_R_supported=False,
            reason='Native classifier is fixed and has no separate disease-query readout argument.',
            branch_definition={
                'original': 'Native original disease queries q.',
                'query': 'Existing native support-query adapter qprime, single original image view.',
                'fixed_aggregation': 'qprime with every visual cross-attention output clamped to the original-q output for the same image; native query residuals, self-attention, FFNs and classifier retained.',
            },
            special_case=('CARZero text-to-image score is reused bitwise from Q; only image-to-text visual updates are clamped.' if model_name == 'CARZero' else None),
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            decoder_source=str(ROOT / DECODER_SOURCES[model_name]),
            decoder_source_sha256=hashlib.sha256((ROOT / DECODER_SOURCES[model_name]).read_bytes()).hexdigest(),
            gradient_training=False, eta=.3, view='identity',
        )
        self.last_audit = {}

    def fit(self, image_paths, labels, class_names, batch_size=8, task='classification'):
        if task != 'classification':
            raise ValueError('This intervention evaluates classification only')
        return self.query.fit(image_paths, labels, class_names, batch_size=batch_size, task=task)

    def _activate(self, class_names):
        if self.model_name == 'MedKLIP':
            self.native._indices(class_names)
        elif self.model_name == 'MAVL':
            self.query._activate(class_names, 'classification')
            if self.native.active_mode != 'feature':
                raise ValueError('MAVL intervention is audited for native feature-mode classification only')

    @torch.inference_mode()
    def predict_all(self, image_paths, class_names, verify=False):
        if not image_paths:
            return {k: np.empty((0, len(class_names)), np.float32)
                    for k in ('original', 'fixed_aggregation', 'query')}
        self._activate(class_names)
        if self.model_name == 'CARZero':
            return self._carzero(image_paths, class_names, verify)
        tape = VisualUpdateTape(self.native.model.decoder)
        with _shared_visual_encoding(self.native, self.model_name):
            with tape.record():
                original = self.query.predict(image_paths, class_names, refined=False, view='identity')
            with tape.replay():
                fixed = self.query.predict(image_paths, class_names, refined=True, view='identity')
            query = self.query.predict(image_paths, class_names, refined=True, view='identity')
            audit = tape.audit()
            if verify:
                with tape.replay():
                    fixed_q = self.query.predict(image_paths, class_names, refined=False, view='identity')
                audit['F_of_q_equals_Original'] = _check_equal(fixed_q, original, 'F(q)=Original')
                audit['Original_native'] = _check_equal(
                    original, self.native.predict(image_paths, class_names), 'Original native')
                audit['Q_existing_adapter'] = _check_equal(
                    query, self.query.predict(image_paths, class_names, refined=True, view='identity'),
                    'Q existing query adapter')
        self.last_audit = audit
        return dict(original=original, fixed_aggregation=fixed, query=query)

    def _carzero(self, image_paths, class_names, verify):
        query = self.query
        q_local, original_q, _ = self.native._text(class_names)
        refined_q = query.fits[tuple(class_names)]
        images = query._images(image_paths, 'identity')
        i_local, i_global = self.native.model.image_encoder_forward(images)
        image_memory = query._memory(i_local, i_global)
        text_original = query._memory(q_local, original_q)
        text_refined = query._memory(q_local, refined_q)
        fusion = self.native.model.fusion_module
        outputs = {k: [] for k in ('original', 'fixed_aggregation', 'query')}
        fixed_q_errors = []
        audit = {}
        for index in range(len(image_paths)):
            memory = image_memory[index:index + 1]
            tape = VisualUpdateTape(fusion.decoder)
            with tape.record():
                original_i2t, _ = fusion(memory, original_q, return_atten=True)
            with tape.replay():
                fixed_i2t, _ = fusion(memory, refined_q, return_atten=True)
            query_i2t, _ = fusion(memory, refined_q, return_atten=True)
            global_i = i_global[index:index + 1]
            original_t2i, _ = fusion(text_original, global_i, return_atten=True)
            query_t2i, _ = fusion(text_refined, global_i, return_atten=True)
            def combined(i2t, t2i):
                return (i2t.squeeze(-1) + t2i.squeeze(-1).transpose(1, 0)) / 2
            outputs['original'].append(combined(original_i2t, original_t2i))
            outputs['query'].append(combined(query_i2t, query_t2i))
            outputs['fixed_aggregation'].append(combined(fixed_i2t, query_t2i))
            audit = tape.audit()
            if verify:
                with tape.replay():
                    fixed_original_i2t, _ = fusion(memory, original_q, return_atten=True)
                fixed_q_errors.append(_check_equal(
                    fixed_original_i2t.cpu().numpy(), original_i2t.cpu().numpy(), 'CARZero F(q)=Original'))
        result = {k: torch.cat(v, 0).float().cpu().numpy() for k, v in outputs.items()}
        audit['text_to_image_reused_from_Q'] = True
        if verify:
            audit['F_of_q_equals_Original'] = dict(passed=True,
                max_abs_error=max(x['max_abs_error'] for x in fixed_q_errors))
            audit['Original_native'] = _check_equal(
                result['original'], self.native.predict(image_paths, class_names), 'CARZero Original native')
            audit['Q_existing_adapter'] = _check_equal(
                result['query'], query.predict(image_paths, class_names, refined=True, view='identity'),
                'CARZero Q existing query adapter')
        self.last_audit = audit
        return result


def create_intervention(model_name, native, query_adapter):
    return BaselineIntervention(model_name, native, query_adapter)
