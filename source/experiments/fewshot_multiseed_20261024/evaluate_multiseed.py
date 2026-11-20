#!/usr/bin/env python3
"""CPU fit of unchanged eleven comparators, then unlock labels/boxes for scoring.

All five fits and predictions are committed before any test labels are opened.
Historical adapted-query outputs are never used as a same-support reference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '4')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'experiments/fewshot_native_query'
sys.path.insert(0, str(OLD))
import numpy as np
import torch
from cache_native_common import sha, write_json
from methods import run_score_methods
from simnl_adapter import run_simnl
sys.path.insert(0, str(ROOT/'baselines/local_setup'))
from evaluate_baseline_buffered import classification_metrics

NATIVE = ('original', 'positive_query', 'query', 'view', 'query_view')
ALIASES = {'original':'original', 'query':'query_refined', 'view':'tta_center_original',
           'query_view':'tta_center_refined'}
DISPLAY = {
 'original':'原始模型', 'positive_query':'仅阳性原型修正查询',
 'query':'正负差分修正原生查询 Q', 'view':'仅双视图 V',
 'query_view':'查询修正＋双视图 Q+V', 'prototype_score_only':'原型分类器',
 'prototype_score_mix_oof':'原型分数混合', 'tip_adapter_ovr_oof':'Tip-Adapter OvR',
 'gda_ovr_ks_oof':'GDA KS OvR', 'class_local_difference_score_oof':'同特征正负差分修正分数',
 'simnl_feature_ovr':'SimNL 特征级 OvR 适配版'}


def checked(spec):
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Frozen file changed: ' + spec['path'])
    return Path(spec['path'])


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def verify_cache_artifacts(cache):
    """Authenticate exported arrays before fitting, scoring, or resuming a fit."""
    record = json.loads((cache/'result.json').read_text())
    digests = record.get('artifact_sha256')
    required = {'support.npz', 'test.npz', 'queries.npz', 'native_predictions.npz'}
    if (cache/'native_maps.npz').exists():
        required.add('native_maps.npz')
    if not isinstance(digests, dict) or not required.issubset(digests):
        raise ValueError('Cache result lacks required artifact SHA256 values')
    for name, digest in digests.items():
        if Path(name).name != name or sha(cache/name) != digest:
            raise ValueError('Cache artifact SHA256 mismatch: ' + name)
    return record


def verify_commit(folder, ppath, cache):
    verify_cache_artifacts(cache)
    commit = json.loads((folder/'fit_commit.json').read_text())
    expected = {'protocol_sha256':sha(ppath), 'cache_result_sha256':sha(cache/'result.json'),
                'fit_info_sha256':sha(folder/'fit_info.json'), 'predictions_sha256':sha(folder/'predictions.npz')}
    if any(commit[key] != value for key, value in expected.items()):
        raise ValueError('Committed fit/provenance changed')
    for path, digest in commit['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Committed algorithm source changed: ' + path)
    return commit


def fit_seed(args, ppath, p, cache, output):
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'fit_commit.json').exists():
        verify_commit(output, ppath, cache)
        return
    if (output/'result.json').exists():
        raise ValueError('Completed result lacks a verified fit commit')
    for spec in p['algorithm_sources'].values():
        checked(spec)
    record = verify_cache_artifacts(cache)
    if record['status'] not in ('complete', 'smoke_complete') or record['protocol_sha256'] != sha(ppath):
        raise ValueError('Incomplete/wrong-protocol feature cache')
    if record['model'] != args.model or record.get('dataset', p['dataset']) != args.dataset:
        raise ValueError('Feature cache model/dataset mismatch')
    s, t, q, native = (load_npz(cache/name) for name in
                       ('support.npz', 'test.npz', 'queries.npz', 'native_predictions.npz'))
    classes = p['class_names']
    count = int(record['test_count'])
    paths = p['test_images'][:count]
    if record['status'] == 'complete' and count != p['test_count']:
        raise ValueError('Complete cache does not cover the fixed test')
    if count <= 0 or count > p['test_count']:
        raise ValueError('Invalid cache test count')
    if t['image_paths'].tolist() != paths or native['image_paths'].tolist() != paths:
        raise ValueError('Test cache identities/order differ')
    if s['image_paths'].tolist() != [r['image'] for r in p['support_rows']]:
        raise ValueError('Support cache identities/order differ')
    for arrays in (s, t, q, native):
        if arrays['class_names'].tolist() != classes:
            raise ValueError('Feature cache class order differs')
    for arrays in (t, q, native):
        if 'labels' in arrays:
            raise ValueError('Fit-visible test/query/native cache contains test labels')
    y = np.asarray([r['labels'] for r in p['support_rows']])
    np.testing.assert_array_equal(s['labels'], y)
    common = dict(support_features=s['features'], labels=y, support_logits=s['logits'],
                  groups=p['support_groups'], test_features=t['features'], test_logits=t['logits'],
                  queries_positive=q['positive'], queries_negative=q['negative'], folds=p['folds'])
    print(f"{args.model}/{args.dataset}/seed_{p['support_seed']}: original score grids, shared support-only folds", flush=True)
    controls = run_score_methods(**common, support_local=s['local'], test_local=t['local'])
    simnl = run_simnl(**common, device='cpu', epoch_candidates=p['fixed_grids']['simnl_epochs'])
    fits = {**controls['fit_info'], **simnl['fit_info']}
    scores = {key:native[key] for key in NATIVE}
    scores.update({key:value for key, value in controls['scores'].items() if key != 'native_original'})
    scores.update(simnl['scores'])
    if set(scores) != set(p['methods']):
        raise ValueError('Eleven-method output incomplete')
    for key, values in scores.items():
        if values.shape != (count, len(classes)) or not np.isfinite(values).all():
            raise ValueError('Nonfinite/misaligned prediction: ' + key)
    for key in NATIVE:
        fits[key] = {'method':key, 'selected_parameters':{}, 'selection_status':'fixed_native_intervention_no_selection',
                     'eta':p['eta'] if key in ('positive_query','query','query_view') else None,
                     'crop_fraction':p['crop_fraction'] if key in ('view','query_view') else None,
                     'support_seed':p['support_seed'], 'score_domain':record['score_domain'],
                     'test_labels_accepted':False, 'test_features_used_for_selection':False}
    write_json(output/'fit_info.json', fits)
    np.savez_compressed(output/'predictions.npz', image_paths=t['image_paths'], class_names=np.asarray(classes),
                        group_ids=np.asarray(p['test_groups'][:count]), **scores)
    paths_to_hash = [Path(__file__), OLD/'methods.py', OLD/'simnl_adapter.py',
                     ROOT/'baselines/local_setup/evaluate_baseline_buffered.py']
    commit = {'protocol_sha256':sha(ppath), 'cache_result_sha256':sha(cache/'result.json'),
              'fit_info_sha256':sha(output/'fit_info.json'), 'predictions_sha256':sha(output/'predictions.npz'),
              'source_sha256':{str(path):sha(path) for path in paths_to_hash},
              'test_labels_and_boxes_opened':False, 'support_seed':p['support_seed'],
              'cv_audit':p['support_cv_audit'], 'elapsed_fit_seconds':time.time()-started}
    write_json(output/'fit_commit.json', commit)


def compare_native(p, model, predictions):
    references = p['stability_references'][model]
    path = checked(references['predictions.npz'])
    values = np.load(path, allow_pickle=False)
    paths = predictions['image_paths'].tolist()
    if values['image_paths'].tolist()[:len(paths)] != paths or values['class_names'].tolist() != p['class_names']:
        raise ValueError('Same-seed stability reference identity/order mismatch')
    if int(values['seed']) != p['support_seed']:
        raise ValueError('Wrong support seed in stability reference')
    audit = {}
    for key, old_key in ALIASES.items():
        expected = values[old_key][:len(paths)]
        np.testing.assert_allclose(predictions[key], expected, atol=2e-5, rtol=2e-5)
        audit[key] = {'reference_key':old_key, 'max_abs_error':float(np.max(np.abs(predictions[key]-expected))),
                      'atol':2e-5, 'rtol':2e-5}
    values.close()
    return {'reference':references['predictions.npz'], 'support_seed':p['support_seed'],
            'passed':True, 'branches':audit, 'historical_single_support_adapted_reference_enforced':False}


def spatial_score(p, model, cache, output, paths, smoke):
    if not (cache/'native_maps.npz').exists():
        if model == 'AlphaRAD' and p['dataset'] == 'chestxdet10' and not smoke:
            raise ValueError('AlphaRAD ChestXDet10 requires five native map branches')
        return {}
    if model != 'AlphaRAD' or p['dataset'] != 'chestxdet10':
        raise ValueError('Spatial evaluation only defined for AlphaRAD ChestXDet10')
    # This function is called only after every support-only fit is committed.
    manifest = p['spatial_manifest']
    rows = json.loads(checked(manifest).read_text())['rows']
    rows = [row for row in rows if row['image'] in set(paths)]
    if not smoke and len(rows) != manifest['count']:
        raise ValueError('Incomplete spatial holdout')
    if not rows:
        return {}
    maps = np.load(cache/'native_maps.npz', allow_pickle=False)
    map_paths = maps['image_paths'].tolist()
    if len(map_paths) != len(set(map_paths)) or maps['class_names'].tolist() != p['class_names']:
        raise ValueError('Spatial map identities/class order differ')
    lookup = {image:i for i,image in enumerate(map_paths)}
    if not set(row['image'] for row in rows).issubset(lookup):
        raise ValueError('Spatial maps omit required annotated images')
    if not set(map_paths).issubset(paths):
        raise ValueError('Spatial map identity outside fixed test')
    idx = [lookup[row['image']] for row in rows]
    source = checked(p['algorithm_sources']['spatial_hits.py'])
    spec = importlib.util.spec_from_file_location('_frozen_stability_spatial', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT/'AlphaRAD'))
    from cxrclip.evaluator_utils import gather_pointinggame_statistics, pointing_game_score
    metrics, values = {}, {}
    for key in NATIVE:
        array = maps[key][idx]
        hits, cases, result = module.spatial_hits(array, rows, p['class_names'])
        # The existing four maps have prior author-helper checks. Check all five
        # on the first seed, including the new positive-only native query map.
        if p['support_seed'] == 42:
            stats = gather_pointinggame_statistics(p['class_names'], array,
                     [r['boxes'] for r in rows], [r['class_names_present'] for r in rows],
                     [r['image'] for r in rows], grounding_type='pointing_game')
            expected = pointing_game_score(stats, _print=False)
            np.testing.assert_allclose(result['pointing_game'], expected['Pointing_Game(Avg)'], atol=1e-12, rtol=0)
            result['author_helper_parity'] = True
        values[key+'_hits'], values[key+'_cases'] = hits, cases
        result.update(rows=len(rows), delta_pp=100*(result['pointing_game']-metrics.get('original',result)['pointing_game']))
        metrics[key] = result
    maps.close()
    reference = p['stability_references'][model]['spatial_result.json']
    old = json.loads(checked(reference).read_text())
    if not smoke:
        for key, old_key in ALIASES.items():
            np.testing.assert_allclose(metrics[key]['pointing_game'], old['branches'][old_key]['pointing_game'], atol=1e-12, rtol=0)
            metrics[key]['same_seed_stability_parity'] = True
    groups = dict(zip(p['test_images'], p['test_groups']))
    np.savez_compressed(output/'spatial.npz', image_paths=np.asarray([r['image'] for r in rows]),
                        class_names=np.asarray(p['class_names']),
                        group_ids=np.asarray([groups[r['image']] for r in rows]), **values)
    return metrics


def score_seed(args, ppath, p, cache, output):
    started = time.time()
    commit = verify_commit(output, ppath, cache)
    if (output/'result.json').exists():
        result = json.loads((output/'result.json').read_text())
        if result['fit_info_sha256'] != commit['fit_info_sha256'] or result['predictions_sha256'] != commit['predictions_sha256']:
            raise ValueError('Existing result differs from committed fit')
        return result
    record = json.loads((cache/'result.json').read_text())
    predictions = load_npz(output/'predictions.npz')
    fits = json.loads((output/'fit_info.json').read_text())
    paths = predictions['image_paths'].tolist()
    rows = json.loads(checked(p['test_manifest']).read_text())['rows'][:len(paths)]
    if [r['image'] for r in rows] != paths:
        raise ValueError('Test label identities/order mismatch')
    y = np.asarray([r['labels'] for r in rows])
    native_audit = compare_native(p, args.model, predictions)
    baseline = classification_metrics(y, predictions['original'], p['class_names'])
    methods = {}
    for key in p['methods']:
        metrics = classification_metrics(y, predictions[key], p['class_names'])
        methods[key] = {'name':DISPLAY[key], 'metrics':metrics,
                        'delta_pp':{m:100*(metrics[m]-baseline[m]) for m in ('macro_auroc','macro_auprc')
                                    if baseline[m] is not None and metrics[m] is not None},
                        'score_domain':fits[key]['score_domain'],
                        'spatial_output':'native' if key in NATIVE and (cache/'native_maps.npz').exists() else 'not_evaluated'}
    smoke = record['status'] == 'smoke_complete'
    spatial = spatial_score(p, args.model, cache, output, paths, smoke)
    verify_commit(output, ppath, cache)
    result = {'status':'smoke_complete' if smoke else 'complete', 'model':args.model, 'dataset':args.dataset,
              'support_seed':p['support_seed'], 'seed':p['support_seed'], 'support_count':p['support_count'],
              'test_count':len(paths), 'class_names':p['class_names'], 'methods':methods, 'spatial':spatial,
              'selection':'All five support-only shared-group 5-fold fits committed before test labels or boxes were opened.',
              'support_cv_audit':p['support_cv_audit'], 'same_seed_stability_reference_audit':native_audit,
              'limitations':p['limitations']+['SimNL is an unchanged feature-only binary OvR port, not exact author reproduction.'],
              'protocol_sha256':sha(ppath), 'catalog_sha256':sha(args.protocol), 'cache':str(cache),
              'cache_result_sha256':commit['cache_result_sha256'], 'fit_info_sha256':commit['fit_info_sha256'],
              'predictions_sha256':commit['predictions_sha256'], 'source_sha256':commit['source_sha256'],
              'spatial_sha256':sha(output/'spatial.npz') if spatial else None,
              'elapsed_seconds':commit['elapsed_fit_seconds']+time.time()-started}
    write_json(output/'result.json', result)
    print(json.dumps({'model':args.model, 'dataset':args.dataset, 'seed':p['support_seed'],
                      'status':result['status'], 'methods':len(methods)}), flush=True)
    return result


def run(args):
    torch.set_num_threads(args.threads)
    catalog = json.loads(args.protocol.read_text())
    if catalog['status'] != 'frozen' or catalog['support_seeds'] != [42,43,44,45,46]:
        raise ValueError('Require complete frozen five-support protocol')
    if args.model not in catalog['models']:
        raise ValueError('Model not specified by protocol')
    # Read catalog seed refs directly: dataset-level JSON contains test labels.
    entries = catalog['datasets'][args.dataset]['seeds']
    work = []
    for seed in catalog['support_seeds']:
        ppath = checked(entries[str(seed)])
        p = json.loads(ppath.read_text())
        if p['dataset'] != args.dataset or p['support_seed'] != seed:
            raise ValueError('Seed protocol mismatch')
        work.append((ppath, p, args.cache/f'seed_{seed}', args.output/f'seed_{seed}'))
    for ppath, p, cache, output in work:
        fit_seed(args, ppath, p, cache, output)
    print('All five support fits/predictions committed; unlocking fixed-test labels and spatial boxes.', flush=True)
    results = [score_seed(args, *item) for item in work]
    summary = {'status':'complete' if all(r['status']=='complete' for r in results) else 'smoke_complete',
               'model':args.model, 'dataset':args.dataset, 'support_seeds':catalog['support_seeds'],
               'methods':catalog['methods'], 'protocol_sha256':sha(args.protocol),
               'results':{str(r['support_seed']):{'path':str(args.output/f"seed_{r['support_seed']}"/'result.json'),
                                                'sha256':sha(args.output/f"seed_{r['support_seed']}"/'result.json')}
                          for r in results}}
    write_json(args.output/'result.json', summary)
    write_json(args.output/'status.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--model', choices=('BiomedCLIP','AlphaRAD'), required=True)
    parser.add_argument('--dataset', choices=('rsna_pneumonia','chestxdet10'), required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    args.protocol, args.cache, args.output = args.protocol.resolve(), args.cache.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output/'failure.json', {'error':repr(error), 'traceback':traceback.format_exc()})
        raise
