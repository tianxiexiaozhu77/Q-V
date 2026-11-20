#!/usr/bin/env python3
"""Complete fixed-support RadZero classification coverage without resampling.

All model/scorer, support pooling, query fitting and view transforms are reused
from the frozen native/phase6 implementation. Only dataset plumbing is new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SETUP = ROOT / 'RadZero_i2cl/local_setup'
PHASE6 = ROOT / 'experiments/query_search_20261021_152010/code_phase6'
for p in (SETUP, PHASE6):
    sys.path.insert(0, str(p))
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
    os.environ[key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from torch.utils.data import DataLoader
from batch_data import load_dataset
from batch_metrics import compute_metrics
from infer_batch import ASSETS_DIR, MODEL_DIR, TestImages, CachedInference, checkpoint_identity, load_model, now, write_json
from extract_frozen_features import pool_original_matching, sha256
from overnight_query_refinement import QueryRefinementMatcher, fit_query_directions
from query_candidates import fit_candidates
from view_geometry import apply_view, VIEW_SPECS

DATASETS = ('chexpert', 'shenzhenxray', 'montgomery', 'covidkaggle', 'siim_pneumothorax')
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')


def loaders(paths, processor, dicom, batch_size=16, workers=2):
    dataset = TestImages(paths, Path('/'), processor, dicom)
    options = dict(batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    if workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **options)


def prompt_configuration(class_names):
    """Reuse released disease wording; deterministic extension for new labels."""
    phrases = {'Consolidation': 'Pulmonary Consolidation', 'Edema': 'Pulmonary Edema',
               'Fracture': 'Bone Fracture'}
    prompts = ['There is ' + phrases.get(name, name) for name in class_names]
    return prompts


def load_inputs(dataset):
    source = ROOT / 'baselines/evaluation/query_transfer_20261022_v1' / f'{dataset}.json'
    support = json.loads(source.read_text())
    test_path = Path(support['test_manifest'])
    assert sha256(test_path) == support['test_manifest_sha256']
    for path, digest in support['support_sources'].items():
        assert sha256(path) == digest, path
    test = json.loads(test_path.read_text())
    assert support['class_names'] == test['class_names']
    assert support['count'] == test['count'] == len(test['rows'])
    assert len(support['support_rows']) == support['support_count']
    for row in support['support_rows']:
        assert sha256(row['image']) == row['byte_sha256']
    supports = [dict(row) for row in support['support_rows']]
    tests = [dict(row) for row in test['rows']]
    assert not ({row['image'] for row in supports} & {row['image'] for row in tests})
    for rows in (supports, tests):
        for row in rows:
            row['native_image'] = row['image']
    native_sources = {}
    if dataset == 'siim_pneumothorax':
        native = load_dataset('SIIM', ASSETS_DIR / 'data')
        by_uid = {Path(p).stem: (str((ASSETS_DIR / 'data' / p).resolve()), y.tolist())
                  for p, y in zip(native['image_paths'], native['labels'])}
        assert len(by_uid) == 1704
        for row in tests:
            uid = Path(row['image']).stem
            path, labels = by_uid[uid]
            assert labels == row['labels']
            row['native_image'] = path
        support_csv = next(Path(p) for p in support['support_sources'] if p.endswith('_support.csv'))
        source_rows = {row['image']: row for row in csv.DictReader(support_csv.open())}
        for row in supports:
            path = source_rows[row['image']]['source_image']
            assert Path(path).stem == Path(row['image']).stem
            row['native_image'] = str(Path(path).resolve())
        support_uids = {Path(row['native_image']).stem for row in supports}
        test_uids = {Path(row['native_image']).stem for row in tests}
        assert not support_uids & test_uids
        native_sources = {str(p): sha256(p) for p in native['source_files']}
    for row in supports + tests:
        path = Path(row['native_image'])
        assert path.is_file(), str(path)
        row['native_byte_sha256'] = sha256(path)
    assert not ({row['native_byte_sha256'] for row in supports}
                & {row['native_byte_sha256'] for row in tests}), 'Exact-byte support/test overlap'
    labels = np.asarray([row['labels'] for row in supports], dtype=np.int8)
    assert labels.sum(0).tolist() == support['support_positive_counts']
    assert np.all(labels.sum(0) > 0) and np.all(labels.sum(0) < len(labels))
    return support, supports, tests, {
        str(source): sha256(source), str(test_path): sha256(test_path),
        **support['support_sources'], **native_sources,
    }


def outputs(model, matcher, pixels, queries):
    view_scores = []
    for view in ('identity', 'center80'):
        aligned = model.forward_vision_model(apply_view(pixels, view))['vision_tokens']
        vision = matcher.prepare_vision(aligned)
        view_scores.append([matcher.match_prepared(vision, q)['logits'] for q in queries])
    original, refined = view_scores[0]
    return dict(zip(BRANCHES, (original, refined,
        torch.stack([original, view_scores[1][0]]).mean(0),
        torch.stack([refined, view_scores[1][1]]).mean(0))))


def known_slice_audit(model, processor, tokenizer, output):
    """Reproduce four historical SIIM branches on the exact first 16 rows."""
    from i2cl_data import load_spec
    historic = ROOT / 'experiments/query_search_20261021_152010/runs/phase6/radzero_SIIM'
    record = json.loads((historic / 'result.json').read_text())
    spec = load_spec(Path(record['config']['split_dir']), 'SIIM', 'test')
    paths = [str(Path(spec['data_root']) / p) for p in spec['image_paths'][:16]]
    bundle = np.load(historic / 'queries.npz', allow_pickle=False)
    old_names = bundle['names'].tolist()
    queries = bundle['queries'][[old_names.index('original'), old_names.index('uniform_eta03')]]
    gpu_queries = torch.as_tensor(queries, device='cuda')
    matcher = QueryRefinementMatcher(model, queries, (0., .3))
    pixels, _, _ = next(iter(loaders(paths, processor, True, 16, 0)))
    actual = outputs(model, matcher, pixels.cuda(), gpu_queries)
    checks = {}
    for branch in BRANCHES:
        historic_name = 'uniform_eta03' if branch == 'query_refined' else branch
        prediction = record['candidates'][historic_name]
        path = Path(prediction['directory']) / 'logits.npy'
        assert sha256(path) == prediction['logits_sha256']
        expected = np.load(path, allow_pickle=False)[:16]
        values = actual[branch].cpu().numpy()
        np.testing.assert_allclose(values, expected, atol=2e-5, rtol=0.)
        checks[branch] = dict(passed=True, rows=16,
            max_abs_error=float(np.abs(values - expected).max()),
            bitwise_equal=bool(np.array_equal(values, expected)), atol=2e-5,
            historical_predictions=str(path), historical_sha256=sha256(path))
    cache = Path(record['config']['cache_manifest']).parent
    labels = np.load(cache / 'support_labels.npy', allow_pickle=False)
    features = np.load(cache / 'support_class_local.npy', allow_pickle=False)
    fitted = fit_candidates(features, labels, queries[0])
    np.testing.assert_array_equal(fitted['queries'][1], queries[1])
    checks['historical_query_fit'] = {'passed': True, 'bitwise_equal': True}
    helper = CachedInference(model, tokenizer, spec['prompts'])
    np.testing.assert_allclose(helper.features.cpu().numpy(), queries[0], atol=1e-6, rtol=0.)
    write_json(output / 'historical_equivalence.json', checks)
    return checks


def run_dataset(args, dataset, model, processor, tokenizer, historical_check):
    from sklearn.metrics import average_precision_score
    started = time.time()
    output = args.output / dataset
    output.mkdir(parents=True, exist_ok=False)
    manifest, supports, tests, hashes = load_inputs(dataset)
    class_names = manifest['class_names']
    prompts = prompt_configuration(class_names)
    if args.limit_test:
        tests = tests[:args.limit_test]
    write_json(output / 'input_manifest.json', dict(dataset=dataset, class_names=class_names,
        prompts=prompts, support_rows=supports, test_rows=tests, source_sha256=hashes,
        identity_audit=manifest['identity_audit'], support_policy=manifest['support_policy']))
    helper = CachedInference(model, tokenizer, prompts)
    original_queries = helper.features.cpu().numpy()
    support_features = []
    dicom = dataset == 'siim_pneumothorax'
    checks = {}
    for pixels, _, indices in loaders([r['native_image'] for r in supports], processor, dicom,
                                     args.batch_size, args.workers):
        pixels = pixels.cuda(non_blocking=True)
        aligned = model.forward_vision_model(pixels)['vision_tokens']
        features, logits, error = pool_original_matching(helper.loss, helper.features, aligned)
        support_features.append(features.cpu().numpy())
        if int(indices[0]) == 0:
            checks['official_api'] = helper.verify_reference(pixels[:2])
            checks['pooled_reconstruction_max_error'] = error
    support_features = np.concatenate(support_features)
    support_labels = np.asarray([r['labels'] for r in supports], dtype=np.int8)
    fitted = fit_candidates(support_features, support_labels, original_queries)
    queries = fitted['queries'][:2]
    legacy = fit_query_directions(support_features, support_labels, original_queries, (0., .3))
    np.testing.assert_array_equal(queries, legacy['queries'])
    checks['phase6_query_bitwise_equal'] = True
    np.savez_compressed(output / 'support_features.npz', features=support_features, labels=support_labels)
    np.savez_compressed(output / 'queries.npz', names=np.asarray(['original', 'query_refined']), queries=queries)
    matcher = QueryRefinementMatcher(model, queries, (0., .3))
    gpu_queries = torch.as_tensor(queries, device='cuda')
    logits = {name: np.full((len(tests), len(class_names)), np.nan, np.float32) for name in BRANCHES}
    checkpoint = checkpoint_identity()
    sources = [Path(__file__).resolve(), SETUP / 'infer_batch.py', SETUP / 'extract_frozen_features.py',
               SETUP / 'overnight_query_refinement.py', SETUP / 'batch_metrics.py',
               PHASE6 / 'query_candidates.py', PHASE6 / 'view_geometry.py',
               MODEL_DIR / 'config.json', MODEL_DIR / 'preprocessor_config.json']
    config = dict(model='RadZero', dataset=dataset, task='classification', class_names=class_names,
        prompts=prompts, prompt_policy='Native There is disease template; reused released Pulmonary Consolidation/Edema and Bone Fracture phrases; otherwise exact class name; fixed before inference.',
        checkpoint=checkpoint, source_sha256={str(p): sha256(p) for p in sources},
        source_manifests=hashes, support_count=len(supports), rows=len(tests),
        batch_size=args.batch_size, workers=args.workers, eta=.3, views=['identity', 'center80'],
        view_specs=VIEW_SPECS['center80'], support_views=['identity'],
        precision='float32; TF32 disabled', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch_version=torch.__version__, seed=42, test_labels_used_for_query_fit=False,
        trainable_base_parameters=0, tuning_or_selection=False,
        classification_aggregation='Arithmetic mean of native logits, then float64 sigmoid; ranking metrics retain native logits as in phase6.',
        image_loading='Native RadZero TestImages: DICOM max scaling/inversion for SIIM; PNG/JPEG minmax normalization otherwise; official 518x518 processor.',
        historical_equivalence=str(args.output / 'historical_equivalence.json'),
        is_smoke=bool(args.limit_test), command=sys.argv)
    write_json(output / 'run_config.json', config)
    offset = 0
    for pixels, _, indices in loaders([r['native_image'] for r in tests], processor, dicom,
                                     args.batch_size, args.workers):
        stop = offset + len(pixels)
        np.testing.assert_array_equal(indices.numpy(), np.arange(offset, stop))
        batch = outputs(model, matcher, pixels.cuda(non_blocking=True), gpu_queries)
        for branch, values in batch.items():
            logits[branch][offset:stop] = values.cpu().numpy()
        offset = stop
        write_json(output / 'progress.json', dict(status='running', completed_images=offset,
            total_images=len(tests), updated_at=now()))
        if offset == len(tests) or offset % 128 == 0 or offset <= args.batch_size:
            print(f'{dataset}: {offset}/{len(tests)}', flush=True)
    assert offset == len(tests)
    labels = np.asarray([r['labels'] for r in tests], dtype=np.int8)
    np.save(output / 'labels.npy', labels, allow_pickle=False)
    write_json(output / 'image_paths.json', [r['image'] for r in tests])
    spec = dict(name=dataset, class_names=class_names, labels=labels)
    branches = {}
    for branch, values in logits.items():
        assert np.isfinite(values).all()
        directory = output / branch
        directory.mkdir()
        np.save(directory / 'logits.npy', values, allow_pickle=False)
        probabilities = torch.sigmoid(torch.from_numpy(values).double()).numpy()
        np.save(directory / 'probabilities.npy', probabilities, allow_pickle=False)
        metrics = compute_metrics(spec, values)
        probability_metrics = compute_metrics(spec, probabilities)
        for key in ('macro_auroc', 'macro_auprc'):
            assert metrics[key] == probability_metrics[key], (branch, key)
        ap = [float(average_precision_score(labels[:, i], values[:, i]))
              for i in range(len(class_names)) if labels[:, i].sum()]
        metrics['macro_average_precision'] = float(np.mean(ap)) if ap else None
        metrics['native_sigmoid_ranking_equivalent'] = True
        metrics['logits_sha256'] = sha256(directory / 'logits.npy')
        metrics['probabilities_sha256'] = sha256(directory / 'probabilities.npy')
        metrics['directory'] = str(directory)
        write_json(directory / 'metrics.json', metrics)
        branches[branch] = metrics
    write_json(output / 'verification.json', checks)
    for path, digest in {**hashes, **config['source_sha256']}.items():
        assert sha256(path) == digest, f'Source changed during evaluation: {path}'
    result = dict(status='complete', model='RadZero', dataset=dataset, task='classification',
        rows=len(tests), support_count=len(supports), class_names=class_names, branches=branches,
        config=config, is_smoke=bool(args.limit_test), elapsed_seconds=time.time()-started,
        completed_at=now(), no_network_weight_updates=True, test_labels_used_for_query_fit=False,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
        artifact_sha256={p.name: sha256(p) for p in (output / 'input_manifest.json', output / 'run_config.json',
            output / 'queries.npz', output / 'support_features.npz', output / 'labels.npy',
            output / 'image_paths.json', output / 'verification.json')})
    write_json(output / 'result.json', result)
    write_json(output / 'progress.json', dict(status='complete', completed_images=offset,
        total_images=len(tests), updated_at=now()))
    print(json.dumps(dict(dataset=dataset, status='complete', branches={k: {
        m: v[m] for m in ('macro_auroc','macro_auprc')} for k,v in branches.items()})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', choices=DATASETS, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit-test', type=int, default=0)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('6', '7'):
        parser.error('This worker is exclusively assigned physical GPU 6 or 7')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA device is not available')
    with torch.inference_mode():
        model, processor, tokenizer = load_model(torch.device('cuda'))
        model = model.float().requires_grad_(False).eval()
        historical_check = known_slice_audit(model, processor, tokenizer, args.output)
        for dataset in args.datasets:
            run_dataset(args, dataset, model, processor, tokenizer, historical_check)


if __name__ == '__main__':
    main()
