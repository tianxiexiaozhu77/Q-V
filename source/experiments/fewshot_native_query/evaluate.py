#!/usr/bin/env python3
"""Fit shared-support comparators, then unlock held-out labels for scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from cache_native_common import MEDICAL, sha, write_json
from methods import run_score_methods
from simnl_adapter import run_simnl

sys.path.insert(0, str(MEDICAL / 'baselines/local_setup'))
from evaluate_baseline_buffered import classification_metrics

DISPLAY = {
 'original':'原始模型', 'positive_query':'仅阳性原型修正查询',
 'query':'正负差分修正原生查询 Q', 'view':'仅双视图 V',
 'query_view':'查询修正＋双视图 Q+V', 'prototype_score_only':'原型分类器',
 'prototype_score_mix_oof':'原型分数混合', 'tip_adapter_ovr_oof':'Tip-Adapter OvR',
 'gda_ovr_ks_oof':'GDA KS OvR', 'class_local_difference_score_oof':'同特征正负差分修正分数',
 'simnl_feature_ovr':'SimNL 特征级 OvR 适配版'}

def run(args):
    started = time.time()
    cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Do not overwrite completed evaluations')
    p = json.loads(args.protocol.read_text())
    record = json.loads((cache/'result.json').read_text())
    if record['status'] not in ('complete','smoke_complete') or record['protocol_sha256'] != sha(args.protocol):
        raise ValueError('Incomplete/wrong-protocol feature cache')
    smoke = record['status'] == 'smoke_complete'
    s = dict(np.load(cache/'support.npz',allow_pickle=False))
    t = dict(np.load(cache/'test.npz',allow_pickle=False))
    q = dict(np.load(cache/'queries.npz',allow_pickle=False))
    native = dict(np.load(cache/'native_predictions.npz',allow_pickle=False))
    classes = p['class_names']
    paths = p['test_images'][:record['test_count']]
    if t['image_paths'].tolist() != paths or native['image_paths'].tolist() != paths:
        raise ValueError('Test cache identities/order differ')
    if s['image_paths'].tolist() != [r['image'] for r in p['support_rows']]:
        raise ValueError('Support cache identities/order differ')
    for arr in (s,t,native,q):
        if arr['class_names'].tolist() != classes:
            raise ValueError('Class order differs')
    y_support = np.asarray([r['labels'] for r in p['support_rows']])
    np.testing.assert_array_equal(s['labels'],y_support)
    if 'labels' in t:
        raise ValueError('Test features must not contain test labels')
    print('Fitting score methods with shared support-only folds',flush=True)
    common = dict(support_features=s['features'], labels=y_support,
        support_logits=s['logits'], groups=p['support_groups'], test_features=t['features'],
        test_logits=t['logits'], queries_positive=q['positive'],queries_negative=q['negative'],
        folds=p['folds'])
    tf = run_score_methods(**common,support_local=s['local'],test_local=t['local'])
    print('Fitting feature-only SimNL with shared support-only folds',flush=True)
    simnl = run_simnl(**common,device='cpu',epoch_candidates=p['fixed_grids']['simnl_epochs'])
    fits = {**tf['fit_info'],**simnl['fit_info']}
    scores = {k:native[k] for k in ('original','positive_query','query','view','query_view')}
    scores.update({k:v for k,v in tf['scores'].items() if k != 'native_original'})
    scores.update(simnl['scores'])
    # Commit every selection before opening any test label or box file.
    write_json(args.output/'fit_info.json',fits)
    np.savez_compressed(args.output/'predictions.npz',image_paths=t['image_paths'],
                        class_names=np.asarray(classes),**scores)
    selection_sha = sha(args.output/'fit_info.json')
    print('All support-only decisions saved; loading test labels for final metrics',flush=True)
    test_spec = p['test_manifest']
    if sha(test_spec['path']) != test_spec['sha256']:
        raise ValueError('Frozen test labels changed')
    rows = json.loads(Path(test_spec['path']).read_text())['rows'][:len(paths)]
    if [r['image'] for r in rows] != paths:
        raise ValueError('Evaluation label order mismatch')
    y = np.asarray([r['labels'] for r in rows])
    methods = {}
    baseline = classification_metrics(y,scores['original'],classes)
    for key,values in scores.items():
        if values.shape != y.shape or not np.isfinite(values).all():
            raise ValueError('Nonfinite or misaligned predictions: '+key)
        metrics = classification_metrics(y,values,classes)
        methods[key] = {'name':DISPLAY[key],'metrics':metrics,
            'delta_pp':{m:100*(metrics[m]-baseline[m]) for m in ('macro_auroc','macro_auprc')
                        if baseline[m] is not None and metrics[m] is not None},
            'score_domain':record['score_domain'] if key in native else fits[key]['score_domain'],
            'spatial_output':'native' if key in native and (cache/'native_maps.npz').exists() else 'not_evaluated'}
    spatial = {}
    if (cache/'native_maps.npz').exists():
        sys.path.insert(0,str(MEDICAL/'AlphaRAD'))
        from cxrclip.evaluator_utils import gather_pointinggame_statistics, pointing_game_score
        spec = p['spatial_manifest']
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Spatial holdout changed')
        spatial_rows = json.loads(Path(spec['path']).read_text())['rows']
        lookup = {path:i for i,path in enumerate(paths)}
        spatial_rows = [r for r in spatial_rows if r['image'] in lookup]
        if not smoke and len(spatial_rows) != spec['count']:
            raise ValueError('Incomplete spatial holdout coverage')
        if spatial_rows:
            maps = np.load(cache/'native_maps.npz',allow_pickle=False)
            if maps['image_paths'].tolist() != paths or maps['class_names'].tolist() != classes:
                raise ValueError('Native spatial map identities or class order differ')
            indices = [lookup[r['image']] for r in spatial_rows]
            if sha(spec['reference']['result']) != spec['reference']['sha256']:
                raise ValueError('Frozen spatial reference changed')
            old = json.loads(Path(spec['reference']['result']).read_text())
            aliases = {'original':'original','query':'uniform_eta03','view':'tta_center_original','query_view':'tta_center_refined'}
            for key in ('original','positive_query','query','view','query_view'):
                stats = gather_pointinggame_statistics(classes,maps[key][indices],
                    [r['boxes'] for r in spatial_rows],[r['class_names_present'] for r in spatial_rows],
                    [r['image'] for r in spatial_rows],grounding_type='pointing_game')
                metrics = pointing_game_score(stats,_print=False)
                value = float(metrics['Pointing_Game(Avg)'])
                if not smoke and key in aliases:
                    expected = old['candidates'][aliases[key]]['metrics']['pointing_game']
                    np.testing.assert_allclose(value,expected,atol=1e-10,rtol=0)
                spatial[key] = {'rows':len(spatial_rows),'pointing_game':value,
                    'per_class':metrics,'delta_pp':100*(value-spatial.get('original',{'pointing_game':value})['pointing_game'])}
    if sha(args.output/'fit_info.json') != selection_sha:
        raise ValueError('Fit decisions changed after test labels were opened')
    source_paths = [Path(__file__),Path(__file__).with_name('methods.py'),Path(__file__).with_name('simnl_adapter.py')]
    result = {'status':'smoke_complete' if smoke else 'complete','model':record['model'],
        'dataset':p['dataset'],'support_count':len(y_support),'test_count':len(y),
        'class_names':classes,'methods':methods,'spatial':spatial,
        'selection':'shared support-only group 5-fold CV; test labels loaded only after fit committed',
        'limitations':p['limitations'] + ['SimNL is a feature-only binary OvR port, not exact author reproduction.',
             'One fixed support draw. Test sets previously exposed in exploratory development; not independent confirmation.'],
        'protocol_sha256':sha(args.protocol),'cache':str(cache),'cache_result_sha256':sha(cache/'result.json'),
        'fit_info_sha256':selection_sha,'predictions_sha256':sha(args.output/'predictions.npz'),
        'source_sha256':{str(v):sha(v) for v in source_paths},'elapsed_seconds':time.time()-started}
    write_json(args.output/'result.json',result)
    print(json.dumps({'status':result['status'],'model':record['model'],'dataset':p['dataset'],
                     'methods':len(methods),'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'failure.json',{'error':repr(error),'traceback':traceback.format_exc()})
        raise
