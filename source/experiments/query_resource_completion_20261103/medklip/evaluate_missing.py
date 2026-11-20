#!/usr/bin/env python3
"""Fixed missing-query resource evaluation, reusing MedKLIP's existing Q worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = ROOT / 'baselines/local_setup'
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HERE))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import numpy as np
import torch
from adapter import Adapter, MODEL_ROOT, BASE_PATH, DESCRIPTIONS_PATH, DESCRIPTIONS_SHA, state_digest
from evaluate_baseline_buffered import sha, write, classification_metrics
from evaluate_query_transfer import load_query, paired, check_native

BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
SCORES = (*BRANCHES, 'center_original', 'center_refined')


def inputs(args):
    path = ROOT / 'baselines/evaluation/query_transfer_20261022_v1' / (args.dataset + '.json')
    support = json.loads(path.read_text())
    testpath = Path(support['test_manifest'])
    assert sha(testpath) == support['test_manifest_sha256']
    data = json.loads(testpath.read_text())
    assert data['class_names'] == support['class_names'] and data['task'] == 'classification'
    rows, supports = data['rows'], support['support_rows']
    expected = {'openi': (7294, 90), 'shenzhenxray': (652, 10), 'montgomery': (128, 10)}[args.dataset]
    assert (len(rows), len(supports)) == expected
    assert len({r['image'] for r in rows}) == len(rows)
    assert not {r['image'] for r in supports} & {r['image'] for r in rows}
    for source, expected_sha in support['support_sources'].items():
        assert sha(source) == expected_sha, source
    paths = [r['image'] for r in supports + rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(zip(paths, pool.map(sha, paths)))
    assert all(hashes[r['image']] == r['byte_sha256'] for r in supports)
    assert not {hashes[r['image']] for r in supports} & {hashes[r['image']] for r in rows}
    write(args.output / 'manifest.json', support)
    write(args.output / 'test_manifest.json', data)
    write(args.output / 'image_sha256.json', hashes)
    return support, data, {str(path): sha(path), str(testpath): sha(testpath)}


def snapshot(args, manifest_sources):
    files = [Path(__file__), HERE / 'adapter.py', BASE_PATH, MODEL_ROOT / 'local_setup/query_refinement_adapter.py',
             MODEL_ROOT / 'local_setup/manifest.json', DESCRIPTIONS_PATH]
    files += [COMMON / p for p in ('evaluate_baseline_buffered.py', 'evaluate_query_transfer.py', 'query_refinement_common.py')]
    files += list((MODEL_ROOT / 'Sample_zero-shot_Classification_CXR14/models').glob('*.py'))
    files += [MODEL_ROOT / 'PreTrain_MedKLIP/data_file/observation explanation.json']
    sources = dict(manifest_sources)
    snapshots = {}
    for source in sorted(set(files)):
        source = source.resolve()
        relative = source.relative_to(ROOT)
        target = args.output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source)] = sha(source)
        snapshots[str(target.relative_to(args.output))] = sha(target)
        assert snapshots[str(target.relative_to(args.output))] == sources[str(source)]
    # Frozen checkpoint and BERT asset identities come from the existing release manifest.
    model_manifest = json.loads((MODEL_ROOT / 'local_setup/manifest.json').read_text())
    assets = {}
    for item in [*model_manifest['checkpoints'], *model_manifest['text_assets']]:
        assert sha(item['path']) == item['sha256'], item['path']
        assets[item['path']] = item['sha256']
    write(args.output / 'source_sha256.json', sources)
    write(args.output / 'snapshot_sha256.json', snapshots)
    write(args.output / 'model_asset_sha256.json', assets)
    return sources, snapshots, assets


def baseline_regression(native, args):
    """A supported ChestXray14 scope must preserve the exact original75 protocol."""
    source = ROOT / 'baselines/evaluation/query_transfer_20261022_v1/chest14.json'
    manifest = json.loads(source.read_text())
    test = json.loads(Path(manifest['test_manifest']).read_text())
    rows, classes = test['rows'][:8], test['class_names']
    assert len(native._indices(classes)) == 14 and len(native.classes) == 75
    reference = manifest['baseline_references']['MedKLIP']
    assert sha(reference['result']) == reference['result_sha256']
    oldfile = Path(reference['result']).parent / 'predictions.npz'
    old = np.load(oldfile, allow_pickle=False)
    assert list(old['class_names']) == classes
    paths = [r['image'] for r in rows]
    assert list(old['image_paths'][:8]) == paths
    with torch.inference_mode():
        actual = native.predict(paths, classes)
    expected = old['scores'][:8]
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
    audit = {'passed': True, 'dataset': 'chest14', 'images': 8, 'query_context_count': 75,
        'max_abs_error': float(np.max(np.abs(actual - expected))), 'atol': 2e-4, 'rtol': 2e-4,
        'reference_result': reference, 'reference_predictions_sha256': sha(oldfile),
        'context_note': 'No added queries in this regression. Extended contexts may legitimately change existing class scores through query self-attention.'}
    write(args.output / 'supported_class_regression.json', audit)
    return audit


def fit_recorded(query, native, supports, classes, batch_size):
    """Record the exact features of the unchanged existing support-fit function."""
    method = native.model.image_encoder
    had_override = 'image_encoder' in native.model.__dict__
    features = []
    def record(pixels):
        values = method(pixels)
        features.append(values.mean(1).detach().cpu())
        return values
    native.model.image_encoder = record
    try:
        fit = query.fit([r['image'] for r in supports], [r['labels'] for r in supports], classes,
                        batch_size=batch_size, task='classification')
    finally:
        if had_override:
            native.model.image_encoder = method
        else:
            del native.model.image_encoder
    return fit, torch.cat(features).numpy()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write(args.output / 'status.json', {'status': 'validating_inputs', 'dataset': args.dataset})
    support, data, manifest_sources = inputs(args)
    sources, snapshots, assets = snapshot(args, manifest_sources)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    native = Adapter('cuda')
    regression = baseline_regression(native, args)
    classes, support_rows = data['class_names'], support['support_rows']
    native._indices(classes)
    assert len(native.classes) == (77 if args.dataset == 'openi' else 76)
    before = native.integrity()
    assert before['all_checkpoint_state_values_unchanged'] and before['original_75_queries_bitwise_equal']
    query = load_query('MedKLIP', native)
    query.metadata['insertion'] = 'Replace only requested disease projections; retain original75 plus the fixed dataset-specific missing queries, native decoder/classifier/norms.'
    with torch.inference_mode():
        fit, features = fit_recorded(query, native, support_rows, classes, args.batch_size)
        np.savez_compressed(args.output / 'query_context.npz', base_raw_queries=native.base_queries.cpu().numpy(),
            full_raw_queries=native.model.disease_book.cpu().numpy(),
            full_projected_queries=native.model.disease_embedding_layer(native.model.disease_book).cpu().numpy(),
            context_names=np.asarray(native.classes))
    np.save(args.output / 'support_features.npy', features, allow_pickle=False)
    np.save(args.output / 'support_labels.npy', np.asarray([r['labels'] for r in support_rows], dtype=np.uint8), allow_pickle=False)
    write(args.output / 'query_adapter.json', fit)
    rows = data['rows'][:args.smoke] if args.smoke else data['rows']
    values = {key: [] for key in SCORES}
    original_context = native.model.disease_book.detach().clone()
    progress_time = 0
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        paths = [r['image'] for r in batch]
        with torch.inference_mode():
            identity = paired(query, 'predict', paths, classes, 'identity')
            center = paired(query, 'predict', paths, classes, 'center80')
            if offset == 0:
                parity = check_native(identity['baseline'], native.predict(paths, classes))
                write(args.output / 'native_parity.json', parity)
        current = {'original': identity['baseline'], 'query_refined': identity['refined'],
            'center_original': center['baseline'], 'center_refined': center['refined'],
            'tta_center_original': (np.asarray(identity['baseline']) + np.asarray(center['baseline'])) * .5,
            'tta_center_refined': (np.asarray(identity['refined']) + np.asarray(center['refined'])) * .5}
        assert torch.equal(native.model.disease_book, original_context)
        for key, value in current.items():
            assert value.shape == (len(batch), len(classes)) and np.isfinite(value).all()
            assert np.all((value >= 0) & (value <= 1))
            values[key].append(value)
        if time.time() - progress_time > 15 or offset + len(batch) == len(rows):
            progress = {'status': 'running', 'dataset': args.dataset, 'done': offset + len(batch),
                        'total': len(rows), 'elapsed_seconds': time.time() - started}
            write(args.output / 'status.json', progress)
            print(json.dumps(progress), flush=True)
            progress_time = time.time()
    labels = np.asarray([r['labels'] for r in rows], dtype=np.uint8)
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    np.savez_compressed(args.output / 'predictions.npz', labels=labels,
        image_paths=np.asarray([r['image'] for r in rows]), class_names=np.asarray(classes), **arrays)
    metrics = {key: classification_metrics(labels, arrays[key], classes) for key in BRANCHES}
    integrity = native.integrity()
    assert integrity['all_checkpoint_state_values_unchanged'] and integrity['original_75_queries_bitwise_equal']
    assert not any(p.requires_grad for p in native.model.parameters())
    assert all(sha(path) == digest for path, digest in sources.items())
    write(args.output / 'extension_integrity.json', integrity)
    artifacts = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'status.json'}
    result = {'status': 'smoke_complete' if args.smoke else 'complete', 'model': 'MedKLIP',
        'dataset': args.dataset, 'task': 'classification', 'rows': len(rows), 'support_count': len(support_rows),
        'class_names': classes, 'is_smoke': bool(args.smoke), 'eta': .3, 'branches': metrics,
        'source_sha256': sources, 'artifact_sha256': artifacts, 'snapshot_sha256': snapshots,
        'model_asset_sha256': assets, 'frozen_descriptions_sha256': DESCRIPTIONS_SHA,
        'native_parity': parity, 'supported_class_regression': regression,
        'extension_integrity': integrity, 'metadata': query.metadata, 'original_adapter_metadata': native.metadata,
        'test_labels_used_for_query_fit': False, 'no_gradient_training': True,
        'all_four_branches_share_context': True, 'query_context_unchanged_during_inference': True,
        'elapsed_seconds': time.time() - started, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'argv': sys.argv, 'fusion': 'arithmetic mean of native positive-class probabilities'}
    write(args.output / 'result.json', result)
    write(args.output / 'status.json', {'status': result['status'], 'dataset': args.dataset, 'done': len(rows)})
    print(json.dumps({key: {m: score[m] for m in ('macro_auroc', 'macro_auprc')} for key, score in metrics.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('openi', 'shenzhenxray', 'montgomery'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists() and not (args.output / 'result.json').exists():
            write(args.output / 'failure.json', {'status': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
