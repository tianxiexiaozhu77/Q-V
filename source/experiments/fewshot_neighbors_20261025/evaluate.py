#!/usr/bin/env python3
"""Fit all neighbors using support-only CV; commit both-view scores without test labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/fewshot_native_query'
sys.path.insert(0, str(OLD))
import numpy as np
from cache_native_common import sha, write_json
from methods import run_score_methods
from simnl_adapter import run_simnl
sys.path.insert(0, str(HERE))
from prepare_protocol import load_protocol
from neighbor_methods import run_neighbor_methods

SEEDS = (42, 43, 44, 45, 46)
TASKS = ('slvh', 'dlv', 'composite_slvh_dlv')
SCORE_METHODS = {'prototype_score_only': 'prototype',
                 'prototype_score_mix_oof': 'prototype_mix',
                 'tip_adapter_ovr_oof': 'tip_adapter', 'gda_ovr_ks_oof': 'gda'}
COMPETITORS = ('prototype', 'prototype_mix', 'tip_adapter', 'gda', 'simnl', 'project_mix', 'linear_probe')
METHODS = ('original', 'query') + COMPETITORS
ALL_METHODS = METHODS + tuple(k + '_view' for k in METHODS)
NATIVE = ('original', 'query', 'original_view', 'query_view')


def now():
    return datetime.now(timezone.utc).isoformat()


def arrays(path):
    with np.load(path, allow_pickle=False) as value:
        result = {k: value[k] for k in value.files}
    if any(k in result for k in ('labels', 'test_labels', 'targets')):
        raise ValueError('Fit-visible cache unexpectedly contains labels: ' + str(path))
    return result


def verify_cache(folder, model):
    record = json.loads((folder / 'result.json').read_text())
    if record['model'] != model or record['status'] not in ('complete', 'smoke_complete'):
        raise ValueError('Incomplete or wrong-model cache')
    digests = record.get('artifact_sha256', {})
    required = {'support_pool.npz', 'test_features.npz', 'queries.npz'}
    required.update(f'seed_{s}/native_predictions.npz' for s in SEEDS)
    if not required <= set(digests):
        raise ValueError('Cache lacks required artifact hashes')
    for filename, digest in digests.items():
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != digest:
            raise ValueError('Cache artifact changed: ' + filename)
    return record


def fit_task(*, support_features, support_logits, labels, groups, folds,
             queries_positive, queries_negative, features_identity, features_center,
             logits_identity, logits_center, support_seed):
    """One fitted predictor per method, applied to both views; no test labels accepted."""
    n = len(features_identity)
    if features_identity.shape != features_center.shape or logits_identity.shape != logits_center.shape:
        raise ValueError('Views must cover identical test examples')
    tx = np.concatenate((features_identity, features_center), axis=0)
    tz = np.concatenate((logits_identity, logits_center), axis=0)
    common = dict(support_features=support_features, labels=labels, groups=groups,
                  folds=folds, test_features=tx, queries_positive=queries_positive,
                  queries_negative=queries_negative)
    start = time.perf_counter()
    # This legacy function accepts a local tensor even for global-only methods.
    # The requested methods never read it; no class-local score control is run.
    classic = run_score_methods(**common, support_logits=support_logits, test_logits=tz,
                               support_local=support_features[:, None, :],
                               test_local=tx[:, None, :], methods=tuple(SCORE_METHODS))
    classic_seconds = time.perf_counter() - start
    start = time.perf_counter()
    simnl = run_simnl(**common, support_logits=support_logits, test_logits=tz,
                      device='cpu', seed=support_seed, epoch_candidates=(0, 10, 20, 40))
    simnl_seconds = time.perf_counter() - start
    start = time.perf_counter()
    neighbor = run_neighbor_methods(**common)
    neighbor_seconds = time.perf_counter() - start
    values = {SCORE_METHODS[k]: v for k, v in classic['scores'].items()}
    values['simnl'] = simnl['scores']['simnl_feature_ovr']
    values.update(neighbor['scores'])
    info = {SCORE_METHODS[k]: v for k, v in classic['fit_info'].items()}
    info['simnl'] = simnl['fit_info']['simnl_feature_ovr']
    neighbor_common = {k: v for k,v in neighbor['fit_info'].items() if k != 'methods'}
    info.update({k: {**neighbor_common, **v} for k,v in neighbor['fit_info']['methods'].items()})
    if set(values) != set(COMPETITORS) or set(info) != set(COMPETITORS):
        raise ValueError('Expected all seven competitors')
    scores = {}
    for name, pred in values.items():
        pred = np.asarray(pred)
        if pred.shape != (2*n, 1) or not np.isfinite(pred).all():
            raise ValueError('Invalid predictions: ' + name)
        scores[name] = pred[:n, 0]
        scores[name + '_view'] = (pred[:n, 0].astype(np.float64) + pred[n:, 0]) * .5
        info[name].update(test_view_policy='one support-original fit; equal identity/center score average',
                          test_views=2, support_views=1, dual_view_parameters_identical=True,
                          validation_views=1, test_labels_received=False)
    timing = dict(classic_four_methods_fit_and_predict_seconds=classic_seconds,
                  simnl_fit_and_predict_seconds=simnl_seconds,
                  neighbor_two_methods_fit_and_predict_seconds=neighbor_seconds,
                  timing_scope='CPU support CV plus final fit and both test-view predictions; excludes frozen encoders')
    return scores, info, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--model', choices=('BiomedCLIP', 'AlphaRAD'), required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    import torch
    torch.set_num_threads(1)
    args.cache = args.cache.resolve()
    args.output = args.output.resolve()
    p = load_protocol(args.protocol, include_test_labels=False)
    rec = verify_cache(args.cache, args.model)
    if rec['protocol_sha256'] != sha(args.protocol):
        raise ValueError('Cache protocol differs')
    pool = arrays(args.cache / 'support_pool.npz')
    test = arrays(args.cache / 'test_features.npz')
    queries = arrays(args.cache / 'queries.npz')
    n = len(test['image_paths'])
    identity = p['test_identity']
    if test['image_paths'].tolist() != identity['image_paths'][:n] or test['patient_ids'].tolist() != identity['patient_ids'][:n]:
        raise ValueError('Test identities/order differ')
    if test['task_keys'].tolist() != list(TASKS):
        raise ValueError('Class/task order changed')
    if rec['status'] == 'complete' and n != 7407:
        raise ValueError('Incomplete full holdout')
    lookup = {str(Path(v).resolve()): i for i, v in enumerate(pool['image_paths'].tolist())}
    if len(lookup) != len(pool['image_paths']):
        raise ValueError('Duplicate pool feature identities')
    args.output.mkdir(parents=True, exist_ok=False)
    started = now()
    sources = {str(f): sha(f) for f in (Path(__file__).resolve(), HERE/'neighbor_methods.py',
               HERE/'prepare_protocol.py', OLD/'methods.py', OLD/'simnl_adapter.py')}
    status = dict(status='fitting', model=args.model, started_at=started,
                  test_labels_opened=False, completed_seeds=[])
    write_json(args.output/'status.json', status)
    for seed in SEEDS:
        native = arrays(args.cache/f'seed_{seed}/native_predictions.npz')
        if native['image_paths'].tolist() != test['image_paths'].tolist():
            raise ValueError('Native/test order mismatch')
        values = {k: np.asarray(native[k]) for k in NATIVE}
        values.update({k: np.empty((n,3),dtype=np.float64) for k in ALL_METHODS if k not in NATIVE})
        fit_info = {}
        for j, task in enumerate(TASKS):
            support = p['support_tasks'][seed][task]
            rows = support['rows']
            ids = [lookup[str(Path(r['image_path']).resolve())] for r in rows]
            expected_pids = [r['patient_id'] for r in rows]
            if pool['patient_ids'][ids].tolist() != expected_pids:
                raise ValueError('Support patient identities differ')
            labels = np.asarray([r['label'] for r in rows],dtype=np.int64)[:,None]
            print(f'{args.model} seed {seed} {task}: fitting seven methods with shared support folds',flush=True)
            score, info, timing = fit_task(support_features=pool['features'][ids],
                support_logits=pool['logits'][ids,j:j+1], labels=labels,
                groups=support['groups'], folds=support['folds'],
                queries_positive=queries['positive'][j:j+1], queries_negative=queries['negative'][j:j+1],
                features_identity=test['features_identity'], features_center=test['features_center'],
                logits_identity=test['logits_identity'][:,j:j+1], logits_center=test['logits_center'][:,j:j+1],
                support_seed=seed)
            for k, pred in score.items(): values[k][:,j] = pred
            fit_info[task] = dict(support_images=[r['image_path'] for r in rows], support_patient_ids=expected_pids,
                support_labels=labels[:,0].tolist(), folds=support['folds'], methods=info, timing=timing,
                support_feature='frozen global projected visual feature',
                native_query_support_feature='original-query class-local' if args.model=='AlphaRAD' else 'global visual',
                same_representation_for_all_methods=False if args.model=='AlphaRAD' else True,
                query=dict(eta=.3,selected_parameters={},trainable_parameters=0,selection='fixed before this batch'))
        if set(values) != set(ALL_METHODS): raise ValueError('Incomplete method suite')
        if any(v.shape!=(n,3) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError('Invalid full prediction matrix')
        folder=args.output/f'seed_{seed}'
        folder.mkdir()
        write_json(folder/'fit_info.json',fit_info)
        np.savez_compressed(folder/'predictions.npz',image_paths=test['image_paths'], patient_ids=test['patient_ids'],
                            task_keys=np.asarray(TASKS),class_names=np.asarray(p['classes']),seed=seed,**values)
        commit=dict(status='committed',model=args.model,seed=seed,test_labels_opened=False,committed_at=now(),
            protocol_sha256=sha(args.protocol),cache_result_sha256=sha(args.cache/'result.json'),
            predictions_sha256=sha(folder/'predictions.npz'),fit_info_sha256=sha(folder/'fit_info.json'),sources=sources)
        write_json(folder/'fit_commit.json',commit)
        status['completed_seeds'].append(seed)
        write_json(args.output/'status.json',status)
    verify_cache(args.cache,args.model)
    if any(sha(f)!=h for f,h in sources.items()): raise ValueError('Algorithm changed during fit')
    status.update(status='smoke_complete' if rec['status']=='smoke_complete' else 'complete',finished_at=now(),
                  task_seed_fits=15,branch_count=18,task_seed_branch_results=270,sources=sources,
                  protocol_sha256=sha(args.protocol),cache_result_sha256=sha(args.cache/'result.json'))
    write_json(args.output/'status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
