#!/usr/bin/env python3
"""Complete two frozen AlphaRAD classification scopes without test-set selection.

Reuse the released native image decoder, preprocessing, text prompts, FLaS,
support-only eta=.3 adaptation, and phase-6 center80/logit aggregation functions.
The input manifests and historical runs are read only. Output directories must
be new. All four branches and every prediction/label are retained for audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
ALPHA = ROOT / 'AlphaRAD'
PHASE6 = ROOT / 'experiments/query_search_20261021_152010/code_phase6'
sys.path[:0] = [str(ALPHA / 'local_setup'), str(ALPHA), str(PHASE6)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

from eval_pretrained import ASSETS, sha256, write_json

CHECKPOINT_SHA = '97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9'
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
HISTORICAL = ROOT / 'experiments/query_search_20261021_152010/runs/phase6/alpharad_siim_pneumothorax'


def save_csv(path, rows, classes):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('image', 'class', 'label'))
        writer.writeheader()
        for row in rows:
            writer.writerow({'image': row['image'],
                'class': repr([c for c, y in zip(classes, row['labels']) if y]),
                'label': repr(row['labels'])})


def prepare(args):
    """Translate frozen rows to native CSVs, without sampling/reordering them."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    source = ROOT / 'baselines/evaluation/matched_phase6_20261022' / (args.dataset + '.json')
    manifest = json.loads(source.read_text())
    assert sha256(manifest['source']['csv']) == manifest['source']['sha256']
    split_name = 'OpenI' if args.dataset == 'openi' else 'SIIM'
    support_csv = ROOT / 'RadZero_i2cl/local_assets/splits/native_public5_local_seed42' / split_name / 'support.csv'
    raw_support = list(csv.DictReader(support_csv.open()))
    classes, test = manifest['class_names'], manifest['rows']
    assert len(test) == manifest['count'] == {'openi': 7294, 'siim_radzero_holdout': 1694}[args.dataset]
    support = []
    for row in raw_support:
        raw_image = ROOT / 'data/radzero/prepared' / row['image_path']
        # SIIM uses the same verified author-converted PNGs as the matched test.
        image = (ASSETS / 'data_prepared/author_default512/SIIM' / (row['image_id'] + '.png')
                 if split_name == 'SIIM' else raw_image)
        if sha256(raw_image) != row['byte_sha256']:
            raise ValueError('Original support image bytes changed: ' + str(raw_image))
        support.append({'image': str(image), 'labels': json.loads(row['labels_json']),
            **{k: row[k] for k in ('image_id', 'group', 'patient_id', 'study_id', 'anchor_class')},
            'raw_image': str(raw_image), 'raw_image_sha256': row['byte_sha256']})
    assert len(support) == (90 if split_name == 'OpenI' else 10)
    for rows in (support, test):
        assert len({r['image'] for r in rows}) == len(rows)
        assert all(len(r['labels']) == len(classes) and set(r['labels']) <= {0, 1} for r in rows)
        assert all(Path(r['image']).is_file() for r in rows)
    assert not {r['image'] for r in support} & {r['image'] for r in test}
    overlaps = {}
    for key in ('group', 'patient_id', 'study_id'):
        a, b = ({r.get(key) for r in rows if r.get(key)} for rows in (support, test))
        overlaps[key] = {'support_known': len(a), 'test_known': len(b), 'overlap': len(a & b)}
        assert not a & b, (key, a & b)
    paths = [r['image'] for r in support + test]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(sha256, paths))
    assert not set(hashes[:len(support)]) & set(hashes[len(support):])
    write_json(args.output / 'image_hashes.json', dict(zip(paths, hashes)))
    for name, rows in [('support', support), ('test', test)]:
        save_csv(args.output / (name + '.csv'), rows, classes)
        write_json(args.output / (name + '_manifest.json'), {'class_names': classes, 'rows': rows})
    audit = {'dataset': args.dataset, 'support_count': len(support), 'evaluation_count': len(test),
        'class_names': classes, 'test_positive_counts': np.asarray([r['labels'] for r in test]).sum(0).tolist(),
        'support_positive_counts': np.asarray([r['labels'] for r in support]).sum(0).tolist(),
        'exact_image_path_overlap': 0, 'exact_image_byte_overlap': 0, 'linkage_checks': overlaps,
        'source_test_manifest': str(source), 'source_test_manifest_sha256': sha256(source),
        'source_test_csv': manifest['source']['csv'], 'source_test_csv_sha256': manifest['source']['sha256'],
        'source_support_csv': str(support_csv), 'source_support_csv_sha256': sha256(support_csv),
        'derived_file_sha256': {f: sha256(args.output / f) for f in
            ('support.csv', 'test.csv', 'support_manifest.json', 'test_manifest.json', 'image_hashes.json')},
        'notes': manifest['notes'], 'sampling': 'exact frozen rows; no new sampling or test-label selection'}
    write_json(args.output / 'data_audit.json', audit)
    return classes, support, test, audit


def four_scores(matcher, pixels):
    """One native forward per view, shared by the two fixed query sets."""
    from view_geometry import apply_view, aggregate_outputs
    native = {}
    for view in ('identity', 'center80'):
        vision = matcher.encode_vision(apply_view(pixels, view))
        native[view] = [matcher.match_prepared(vision, q) for q in matcher.queries]
    return {
        'original': native['identity'][0]['logits'],
        'query_refined': native['identity'][1]['logits'],
        'tta_center_original': aggregate_outputs({v: native[v][0] for v in native}, list(native))['logits'],
        'tta_center_refined': aggregate_outputs({v: native[v][1] for v in native}, list(native))['logits'],
    }


def historical_check(evaluator, args, loader_options):
    """Reproduce all four historical SIIM branches on the unchanged first eight images."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from cxrclip.data.datasets.image_classification import ImageClassificationDataset
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator

    paths = json.loads((HISTORICAL / 'image_paths.json').read_text())[:8]
    labels = np.load(HISTORICAL / 'labels.npy', allow_pickle=False)[:8]
    reference_rows = [{'image': p, 'labels': y.tolist()} for p, y in zip(paths, labels)]
    check_csv = args.output / 'historical_check.csv'
    save_csv(check_csv, reference_rows, ['Pneumothorax'])
    existing = evaluator.data_loader_dict[args.dataset].dataset
    dataset = ImageClassificationDataset('siim_pneumothorax', str(check_csv), 'test', existing.normalize)
    dataset.image_transforms = existing.image_transforms
    loader = DataLoader(dataset, collate_fn=dataset.collate_fn, **loader_options)
    q0, _ = original_queries_from_evaluator(evaluator, ['Pneumothorax'])
    saved_q0 = np.load(HISTORICAL / 'candidates/original/query.npy')
    np.testing.assert_allclose(q0, saved_q0, rtol=1e-5, atol=2e-6)
    q1 = np.load(HISTORICAL / 'candidates/uniform_eta03/query.npy')
    matcher = QueryRefinementMatcher(evaluator.model, np.stack([q0, q1]), (0., .3),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    actual = {key: [] for key in BRANCHES}
    with torch.inference_mode():
        for batch in loader:
            for key, values in four_scores(matcher, batch['images'].to(matcher.device)).items():
                actual[key].append(torch.sigmoid(values).cpu().numpy())
    errors, sources = {}, {}
    for key in BRANCHES:
        old_key = 'uniform_eta03' if key == 'query_refined' else key
        prediction_file = HISTORICAL / 'candidates' / old_key / 'probabilities.npy'
        expected = np.load(prediction_file, allow_pickle=False)[:8]
        values = np.concatenate(actual[key])
        np.testing.assert_allclose(values, expected, rtol=1e-5, atol=2e-6)
        errors[key] = float(np.max(np.abs(values - expected)))
        sources[str(prediction_file)] = sha256(prediction_file)
    return {'passed': True, 'images': 8, 'historical_run': str(HISTORICAL),
        'probability_max_abs_errors': errors, 'reference_sha256': sources,
        'note': 'Old support-derived queries used only for regression; new adapters fit their own frozen supports.'}


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.prompt import constants
    from eval_query_refinement import classification_metrics, original_helper_check
    from query_refinement import (fit_query_directions,
        fit_reaggregation_query_directions, fit_uncertainty_query_directions,
        QueryRefinementMatcher, original_queries_from_evaluator)
    from query_candidates import fit_candidates
    from view_geometry import VIEW_SPECS

    start = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'progress.json', {'status': 'preparing', 'dataset': args.dataset})
    classes, support, test, audit = prepare(args)
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Released checkpoint SHA256 mismatch')
    source_files = [Path(__file__).resolve(), ALPHA / 'local_setup/query_refinement.py',
        ALPHA / 'local_setup/eval_query_refinement.py', PHASE6 / 'query_candidates.py', PHASE6 / 'view_geometry.py',
        ALPHA / 'cxrclip/evaluator.py', ALPHA / 'cxrclip/evaluator_utils.py',
        ALPHA / 'cxrclip/data/datasets/image_classification.py', ALPHA / 'cxrclip/data/data_utils.py',
        ALPHA / 'cxrclip/model/radzero.py']
    source_files = [p for p in source_files if p.is_file()]
    provenance = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': CHECKPOINT_SHA,
        'source_sha256': {str(p): sha256(p) for p in source_files},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'argv': sys.argv,
        'eta': .3, 'method': args.method, 'views': {v: VIEW_SPECS[v] for v in ('identity', 'center80')},
        'aggregation': 'arithmetic mean of native logits, then sigmoid',
        'test_labels_used_for_query_fit': False, 'selection_performed': False, 'weight_updates': False}
    write_json(args.output / 'provenance.json', provenance)
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    setattr(constants, args.dataset.upper(), classes)
    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        opts['prefetch_factor'] = 2
    data = {name: {'name': name, 'data_type': 'image_classification', 'data_path': str(args.output / csv_name),
                  'n_class': len(classes), 'normalize': None}
            for name, csv_name in ((args.dataset, 'test.csv'), (args.dataset + '_support', 'support.csv'))}
    config = {'base': {'seed': 1234}, 'test': {'checkpoint': [str(args.checkpoint)]},
              'data_test': data, 'dataloader': {'test': opts}}
    write_json(args.output / 'eval_config.json', config)
    evaluator = Evaluator(config, [str(args.checkpoint)], 'test')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint['model'])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    regression = historical_check(evaluator, args, opts)
    write_json(args.output / 'historical_check.json', regression)
    print('Historical four-branch regression passed:', regression['probability_max_abs_errors'], flush=True)
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    extractor = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                      use_all_image_tokens=evaluator.use_all_image_tokens)
    support_features, support_seen, support_vision = [], [], []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict[args.dataset + '_support']:
            if args.method == 'reaggregation':
                prepared = extractor.encode_vision(batch['images'])
                values = extractor.match_prepared(
                    prepared, extractor.queries[0], return_features=True)
                support_vision.append({key: [value.cpu() for value in values]
                                       for key, values in prepared.items()})
            else:
                values = extractor(batch['images'], return_features=True)[0.]
            support_features.append(values['class_local'].cpu().numpy())
            support_seen.extend(batch['image_paths'])
    assert support_seen == [r['image'] for r in support]
    features = np.concatenate(support_features)
    y_support = np.asarray([r['labels'] for r in support], dtype=np.uint8)
    if args.method == 'reaggregation':
        fit = fit_reaggregation_query_directions(
            support_vision, y_support, original, extractor,
            eta=.3, max_iter=args.reaggregation_max_iter,
            query_tol=args.reaggregation_query_tol,
            patience=args.reaggregation_patience)
    elif args.method == 'uncertainty':
        fit = fit_uncertainty_query_directions(features, y_support, original,
            etas=(0., .3), covariance_shrinkage=.2, ridge=1e-3, reliability_power=1.0)
    else:
        fit = fit_query_directions(features, y_support, original, etas=(0., .3))
        phase6_fit = fit_candidates(features, y_support, original, num_heads=evaluator.model.num_radzero_heads)
        for ours, historical in [(0, 'original'), (1, 'uniform_eta03')]:
            np.testing.assert_array_equal(fit['queries'][ours], phase6_fit['queries'][phase6_fit['names'].index(historical)])
    np.save(args.output / 'support_class_local.npy', features, allow_pickle=False)
    fit_arrays = {key: value for key, value in fit.items()
                  if isinstance(value, np.ndarray)}
    np.savez_compressed(args.output / 'query_adapter.npz', **fit_arrays)
    write_json(args.output / 'query_adapter.json', {'eta': .3, 'class_names': classes, 'prompts': prompts,
        'support_positive_counts': fit['positive_counts'].tolist(), 'support_negative_counts': fit['negative_counts'].tolist(),
        'phase6_queries_bitwise_equal': args.method == 'closed_form',
        'adaptation_mode': args.method,
        'reaggregation': ({key: value for key, value in fit.items()
                           if key.startswith('reaggregation_')}
                          if args.method == 'reaggregation' else None),
        'query_sha256': sha256(args.output / 'query_adapter.npz'),
        'support_count': len(support), 'support_views': ['identity'], 'test_labels_used_for_query_fit': False})
    matcher = QueryRefinementMatcher(evaluator.model, fit['queries'], fit['etas'],
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    loader = evaluator.data_loader_dict[args.dataset]
    if args.limit_test:
        test = test[:args.limit_test]
        loader = DataLoader(Subset(loader.dataset, range(len(test))), collate_fn=loader.dataset.collate_fn, **opts)
    y = np.asarray([r['labels'] for r in test], dtype=np.uint8)
    logits, probabilities = {}, {}
    for branch in BRANCHES:
        branch_dir = args.output / 'branches' / branch
        branch_dir.mkdir(parents=True)
        logits[branch] = np.lib.format.open_memmap(branch_dir / 'logits.npy', mode='w+', dtype=np.float32, shape=y.shape)
        probabilities[branch] = np.lib.format.open_memmap(branch_dir / 'probabilities.npy', mode='w+', dtype=np.float32, shape=y.shape)
        logits[branch][:] = np.nan
        probabilities[branch][:] = np.nan
    np.save(args.output / 'labels.npy', y, allow_pickle=False)
    write_json(args.output / 'image_paths.json', [r['image'] for r in test])
    done = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            count = len(batch['images'])
            assert batch['image_paths'] == [r['image'] for r in test[done:done + count]]
            np.testing.assert_array_equal(batch['labels'].numpy(), y[done:done + count])
            if done == 0:
                helper = original_helper_check(evaluator, matcher, batch, classes)
                write_json(args.output / 'native_helper_check.json', helper)
            for branch, values in four_scores(matcher, batch['images'].to(matcher.device)).items():
                logits[branch][done:done + count] = values.cpu().numpy()
                probabilities[branch][done:done + count] = torch.sigmoid(values).cpu().numpy()
            done += count
            if batch_index % 16 == 0 or done == len(test):
                for values in [*logits.values(), *probabilities.values()]:
                    values.flush()
                progress = {'status': 'running', 'images_completed': done, 'images_total': len(test),
                            'elapsed_seconds': time.time() - start}
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    assert done == len(test)
    branches = {}
    for branch in BRANCHES:
        assert np.isfinite(probabilities[branch]).all() and np.isfinite(logits[branch]).all()
        metrics = classification_metrics(y, probabilities[branch])
        metrics['per_class'] = dict(zip(classes, metrics['per_class']))
        files = {kind: str(args.output / 'branches' / branch / (kind + '.npy')) for kind in ('logits', 'probabilities')}
        branches[branch] = {'metrics': metrics, 'files': files,
            'sha256': {kind: sha256(path) for kind, path in files.items()}}
    for path, before in provenance['source_sha256'].items():
        assert sha256(path) == before, 'Code changed during inference: ' + path
    assert not any(p.requires_grad for p in evaluator.model.parameters())
    result = {'status': 'smoke_complete' if args.limit_test else 'complete', 'model': 'AlphaRAD',
        'task': 'classification', 'dataset': args.dataset, 'rows': done, 'support_count': len(support),
        'class_names': classes, 'is_smoke': bool(args.limit_test), 'branches': branches,
        'checkpoint_sha256': CHECKPOINT_SHA, 'data_audit_sha256': sha256(args.output / 'data_audit.json'),
        'labels_sha256': sha256(args.output / 'labels.npy'), 'image_paths_sha256': sha256(args.output / 'image_paths.json'),
        'provenance_sha256': sha256(args.output / 'provenance.json'), 'historical_check': regression,
        'native_helper_check': helper, 'ignored_checkpoint_keys': ignored,
        'no_network_weight_updates': True,
        'phase6_query_fit_bitwise_equal': args.method == 'closed_form',
        'batch_size': args.batch_size, 'elapsed_seconds': time.time() - start,
        'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / 1024**2}
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'progress.json', {'status': result['status'], 'images_completed': done, 'images_total': done})
    print(json.dumps({k: v['metrics'] for k, v in branches.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('openi', 'siim_radzero_holdout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ASSETS / 'model/xraydinov2_mpnet/eccv_mpnet_518.tar')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--method', choices=('closed_form', 'reaggregation', 'uncertainty'), default='closed_form')
    parser.add_argument('--reaggregation-max-iter', type=int, default=3)
    parser.add_argument('--reaggregation-query-tol', type=float, default=1e-4)
    parser.add_argument('--reaggregation-patience', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-test', type=int, default=0)
    args = parser.parse_args()
    if (args.reaggregation_max_iter < 1 or args.reaggregation_max_iter > 32
            or not math.isfinite(args.reaggregation_query_tol) or args.reaggregation_query_tol < 0
            or args.reaggregation_patience < 1
            or args.reaggregation_patience > args.reaggregation_max_iter):
        parser.error('Invalid reaggregation stopping settings')
    args.output = args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.exists() and not (args.output / 'result.json').exists():
            write_json(args.output / 'failure.json', {'status': 'failed', 'type': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
