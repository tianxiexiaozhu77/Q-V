"""Compare protocol output with native forward and actual AlphaRAD binary helper.

No dataset labels or performance-based prompt choices enter this audit.
"""
import ast
from pathlib import Path
import json
from types import SimpleNamespace
import time
import numpy as np
from scipy.special import softmax
import torch
from PIL import Image
import adapter

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
BASELINES = adapter.BASELINES_ROOT
MODEL = ROOT.name
image = json.loads((BASELINES / MODEL / 'local_setup/cpu_smoke.json').read_text())['image']
classes = ['Pneumonia', 'Pneumothorax', 'Pleural_Thickening']
phrases = adapter.disease_phrases(classes)
start = time.time()
a = adapter.create_adapter('cpu')
scores = a.predict([image], classes)
assert scores.shape == (1, 3) and np.isfinite(scores).all()
assert ((scores > 0) & (scores < 1)).all()
report = dict(status='passed', model=MODEL, protocol_revision='alpharad_binary_v1', device='cpu',
              image=image, classes=classes, resolved_phrases=phrases, scores=scores.tolist(),
              metadata=a.metadata, labels_used=False)

with torch.inference_mode():
    if MODEL == 'BiomedCLIP':
        with Image.open(image) as im:
            images = a.preprocess(im.convert('RGB'))[None]
        prompts = [f'There is no {d}.' for d in phrases] + [f'There is {d}.' for d in phrases]
        tokens = a.tokenizer(prompts)
        features, texts, scale = a.model(images, tokens)
        cosine = (features @ texts.T).numpy()
        expected = softmax(np.stack([cosine[:, :3], cosine[:, 3:]], axis=-1), axis=-1)[..., 1]
        report['native_exp_logit_scale_not_used'] = float(scale)
        image_features = features.numpy()
        def encode_text(prompts, unused_length):
            return {'cls_emb': a.model.encode_text(a.tokenizer(prompts), normalize=True).numpy()}
    else:
        cosine_pairs = []
        for disease in phrases:
            cosine_pairs.append([
                a.engine.get_similarity_score_from_raw_data(Path(image), f'There is no {disease}.'),
                a.engine.get_similarity_score_from_raw_data(Path(image), f'There is {disease}.')])
        expected = softmax(np.array(cosine_pairs), axis=-1)[:, 1][None]
        image_features = a.image_engine.get_projected_global_embedding(Path(image)).numpy()[None]
        def encode_text(prompts, unused_length):
            return {'cls_emb': a.text_engine.get_embeddings_from_prompt(prompts, verbose=False).numpy()}
        maps = a.grounding([image], classes)[0]
        native = np.stack([a.engine.get_similarity_map_from_raw_data(Path(image), f'There is {d}.') for d in phrases])
        report['native_grounding_max_abs_diff'] = float(np.nanmax(np.abs(maps - native)))
        report['grounding_shape'] = list(maps.shape)
        assert np.allclose(maps, native, atol=1e-6, rtol=1e-5, equal_nan=True)

    # Execute the actual unchanged AlphaRAD helper body. Disease phrases are already
    # resolved by the fixed protocol; mapping is identity here, with no descriptions.
    source_path = BASELINES.parent / 'AlphaRAD/cxrclip/base_evaluator.py'
    tree = ast.parse(source_path.read_text())
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_zeroshot_binary_helper')
    namespace = {'np': np, 'softmax': softmax, 'disease_name_mapping': lambda s: s,
                 'curate_dqn_input_labels': lambda *args: ''}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), 'exec'), namespace)
    fake = SimpleNamespace(prompt_template='', observation_explanation='', anonymize_observation_explanation=False,
                           config={'base': {'text_max_length': 256}}, similarity_type='cosine', encode_text=encode_text)
    helper_output = namespace['_zeroshot_binary_helper'](fake, image_features, None, list(phrases), list(phrases))[:, 1][None]
    report['alpharad_actual_helper_max_abs_diff'] = float(np.max(np.abs(scores - helper_output)))
    assert np.allclose(scores, helper_output, atol=1e-6, rtol=1e-5)

report['native_forward_binary_max_abs_diff'] = float(np.max(np.abs(scores - expected)))
assert np.allclose(scores, expected, atol=1e-6, rtol=1e-5)
# Other disease labels must not enter this class's softmax denominator.
single = a.predict([image], [classes[0]])
report['class_independence_max_abs_diff'] = float(np.max(np.abs(scores[:, :1] - single)))
assert np.allclose(scores[:, :1], single, atol=1e-6, rtol=1e-5)
report['elapsed_seconds'] = time.time() - start
(ROOT / 'local_setup/cpu_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
