"""CPU validation shared by the two explicit CheXchoNet text-query overlays."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MEDICAL = ROOT.parent


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MedKLIP', 'MAVL'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    overlay = module_at('chexcho_validation_overlay', ROOT / args.model / 'local_setup/chexchonet_adapter.py')
    native = overlay.create_adapter('cpu')
    classes = [item['class_name'] for item in overlay.TASKS]
    support_csv = MEDICAL / 'AlphaRAD/local_setup/chexchonet_20261023/slvh/support.csv'
    with support_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    paths = [row['image'] for row in rows]
    labels = [ast.literal_eval(row['label']) for row in rows]
    with torch.inference_mode():
        native_predictions = native.predict(paths[:2], classes)
        if native_predictions.shape != (2, 3) or not np.isfinite(native_predictions).all():
            raise AssertionError('Native new-query forward failed')
        query_module = module_at('chexcho_validation_query', ROOT / args.model / 'local_setup/query_refinement_adapter.py')
        query = query_module.create_query_adapter(native)
        fit = query.fit(paths, labels, [classes[0]], batch_size=5)
        first = query.predict_pair(paths[:2], [classes[0]], view='identity')
        original = native.predict(paths[:2], [classes[0]])
        if not np.allclose(first['baseline'], original, atol=2e-6, rtol=1e-5):
            raise AssertionError('Query wrapper does not preserve native baseline')
        center = query.predict_pair(paths[:2], [classes[0]], view='center80')
        for arm in (first, center):
            for values in arm.values():
                if values.shape != (2, 1) or not np.isfinite(values).all():
                    raise AssertionError('Invalid query/center prediction')
        state_after = overlay._state_digest(native.model)
        integrity = native.metadata['extension_integrity']
        if state_after != integrity['state_sha256_before']:
            raise AssertionError('Query fit or inference changed checkpoint state')
    result = {
        'status': 'passed', 'model': args.model, 'device': 'cpu',
        'new_query_forward_shape': list(native_predictions.shape),
        'all_three_native_predictions': native_predictions.tolist(),
        'support_counts': {'positive': 5, 'negative': 5},
        'native_parity_max_abs_error': float(np.max(np.abs(first['baseline'] - original))),
        'original_context_and_parameter_integrity': integrity,
        'state_unchanged_after_query_fit_and_both_views': True,
        'eta': fit['eta'], 'fit_task': classes[0],
        'frozen_original_adapter_sha256': hashlib.sha256((ROOT / args.model / 'local_setup/adapter.py').read_bytes()).hexdigest(),
        'overlay_sha256': hashlib.sha256((ROOT / args.model / 'local_setup/chexchonet_adapter.py').read_bytes()).hexdigest(),
        'query_adapter_sha256': hashlib.sha256((ROOT / args.model / 'local_setup/query_refinement_adapter.py').read_bytes()).hexdigest(),
        'prompts_sha256': overlay.PROMPTS_SHA256,
        'not_a_test_metric': True,
    }
    target = ROOT / 'protocols/chexchonet' / args.model / 'local_setup'
    (target / 'cpu_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest_path = target / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status='ready', cpu_validation=str(target / 'cpu_validation.json'),
                    cpu_validation_sha256=hashlib.sha256((target / 'cpu_validation.json').read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (target / 'READY.json').write_text(json.dumps({'status': 'ready', 'ready': True, 'model': args.model,
        'adapter_sha256': hashlib.sha256((target / 'adapter.py').read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256((target / 'python_local.sh').read_bytes()).hexdigest(),
        'manifest': str(manifest_path), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'cpu_validation': str(target / 'cpu_validation.json')}, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
