"""CPU-only parity audit of the author's unseen-COVID query extension."""
from pathlib import Path
import importlib.util
import json
import hashlib
import time
import types
import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT.parent
torch.set_num_threads(4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def dataset(name):
    return json.loads((BASELINES / 'evaluation/matched_phase6_20261022' / (name + '.json')).read_text())


def load_module(name, source=None):
    path = ROOT / 'local_setup/adapter.py'
    module = types.ModuleType(name)
    # Preserve the original repository root for the saved, unmodified adapter.
    module.__file__ = str(path)
    exec(compile(source if source is not None else path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def main():
    start = time.time()
    saved = BASELINES / '.cache/MedKLIP_before_covid_extension/adapter.py'
    old_module = load_module('medklip_before_covid', saved.read_text())
    module = load_module('medklip_with_covid')
    old = old_module.create_adapter('cpu')
    new = module.create_adapter('cpu')
    checks = {}
    native_json = json.loads((BASELINES / 'MAVL/Zero-shot_classification/observation explanation.json').read_text())
    assert module.COVID_DESCRIPTION == native_json['covid19']
    checks['author_appendix_definition_matches_published_native_example'] = 'passed'
    assert old.classes == new.base_classes and len(new.base_classes) == 75
    assert torch.equal(old.model.disease_book, new.base_queries)
    checks['original75_query_embeddings_bitwise_equal'] = 'passed'
    paths = [r['image'] for r in dataset('chestxdet10')['rows'][:2]]
    classes = dataset('chest14')['class_names']
    before = old.predict(paths, classes)
    base = new.predict(paths, classes)
    assert np.array_equal(before, base)
    checks['original14_class_scores_max_abs'] = float(np.abs(before - base).max())
    x = torch.stack([new.transform(Image.open(p).convert('RGB')) for p in paths])
    assert torch.equal(old.model(x), new.model(x))
    checks['original75_full_logits_bitwise_equal'] = 'passed'

    covid_paths = [r['image'] for r in dataset('covidkaggle')['rows'][:2]]
    actual = new.predict(covid_paths, ['Covid'])
    assert new.classes == old.classes + ['covid19']
    assert torch.equal(new.model.disease_book[:75], new.base_queries)
    checks['covid_context_preserves_first75_embeddings'] = 'passed'

    # Construct the unmodified author model with all 76 descriptions, exactly
    # following MedKLIP Section3.5 and the released native COVID example.
    texts = list(new.book.values()) + [module.COVID_DESCRIPTION]
    tokens = new.tokenizer(texts, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    config = dict(d_model=256, res_base_model='resnet50', H=4, N=4, dropout=.1,
                  attribute_set_size=2, num_queries=75,
                  text_encoder=str(ROOT / 'local_assets/Bio_ClinicalBERT'))
    torch.manual_seed(42)
    native = module.MedKLIP(config, tokens)
    state = torch.load(new.checkpoint, map_location='cpu')['model']
    native.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    native.eval().requires_grad_(False)
    x = torch.stack([new.transform(Image.open(p).convert('RGB')) for p in covid_paths])
    reference = native(x).softmax(-1)[:, 75:76, 1].numpy()
    error = float(np.abs(reference - actual).max())
    assert error < 2e-6, error
    checks['native76_classification_max_abs'] = error
    checks['native76_strict_load_and_score_reference'] = 'passed'

    # Cover the two new tasks and a rectangular real image for axis/extent checks.
    geometry_paths = [covid_paths[0], dataset('qatacovid_seg')['rows'][0]['image'],
                      dataset('shenzhenxray')['rows'][0]['image']]
    geometry = []
    for path in geometry_paths:
        with Image.open(path) as im:
            size = (im.height, im.width)
            x = new.transform(im.convert('RGB'))[None]
        captured = []
        hook = native.decoder.register_forward_hook(lambda model, args, out: captured.append(out[1]))
        try:
            native(x)
        finally:
            hook.remove()
        ws = captured[0]
        attention = (ws[-4] + ws[-3] + ws[-2] + ws[-1]) / 4
        reference = attention[:, 75:76].reshape(1, 1, 14, 14)
        reference = reference.repeat_interleave(16, -2).repeat_interleave(16, -1)
        reference = torch.nn.functional.interpolate(reference, size=size, mode='nearest')[0].numpy()
        actual_map = new.grounding([path], ['covid'])[0]
        error = float(np.abs(reference - actual_map).max())
        assert actual_map.shape == (1, *size) and np.isfinite(actual_map).all()
        assert error < 2e-7, error
        geometry.append({'image': path, 'shape': list(actual_map.shape), 'max_abs': error})
    checks['native76_grounding_reference_and_original_coordinates'] = geometry
    restored = new.predict(paths, classes)
    assert new.active_mode == 'base75' and len(new.classes) == 75
    assert new.model.disease_book is new.base_queries
    assert np.array_equal(before, restored)
    checks['base75_scores_bitwise_equal_after_covid_roundtrip'] = 'passed'
    for name in ['Tuberculosis', 'Granuloma', 'Calcified Granuloma']:
        try:
            new.check_classes([name])
        except NotImplementedError:
            continue
        raise AssertionError('Unverified query accepted: ' + name)
    checks['unverified_definitions_remain_unsupported'] = 'passed'
    checkpoint_sha = sha(new.checkpoint)
    assert checkpoint_sha == 'b44748e00cae76f60b76a3f35d2459527fdb7624562f393de05b89c3174f8de4'
    result = dict(status='passed', device='cpu', elapsed_seconds=time.time() - start,
                  adapter_sha256=sha(ROOT / 'local_setup/adapter.py'),
                  original_adapter_sha256=sha(saved), checkpoint_sha256=checkpoint_sha,
                  checks=checks)
    (ROOT / 'local_setup/covid_extension_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
