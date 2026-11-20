#!/usr/bin/env python3
"""Freeze existing paired samples and support-only grouped CV before inference."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def make_folds(labels, groups, seed=42, nfolds=5):
    """Select label-balanced group folds from support only, never test outcomes."""
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if len(unique) < nfolds:
        raise ValueError('Too few independent support groups')
    rng = np.random.default_rng(seed)
    best, score = None, float('inf')
    # Fixed search budget concerns support label balance, not model performance.
    for _ in range(2000):
        partitions = np.array_split(rng.permutation(unique), nfolds)
        folds, balance = [], 0.0
        for part in partitions:
            val = np.flatnonzero(np.isin(groups, part))
            train = np.flatnonzero(~np.isin(groups, part))
            pos = labels[train].sum(0)
            if np.any(pos == 0) or np.any(pos == len(train)):
                break
            target = labels.sum(0) / nfolds
            balance += float(np.square((labels[val].sum(0)-target)/(target+1)).sum())
            balance += abs(len(val)-len(labels)/nfolds) / len(labels)
            folds.append({'train': train.tolist(), 'validation': val.tolist()})
        if len(folds) == nfolds and balance < score:
            best, score = folds, balance
    if best is None:
        raise ValueError('No 5-fold grouped partition with both classes in every training fold')
    coverage = sorted(i for f in best for i in f['validation'])
    assert coverage == list(range(len(labels)))
    for fold in best:
        assert not set(groups[fold['train']]) & set(groups[fold['validation']])
    return best


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog_path = ROOT/'baselines/evaluation/query_transfer_20261022_v1/catalog.json'
    catalog = json.loads(catalog_path.read_text())['datasets']
    aggregate = json.loads((ROOT/'baselines/QUERY_REFINEMENT_RESULTS.json').read_text())
    files = {}
    for key in ['rsna_pneumonia', 'chestxdet10']:
        e = catalog[key]
        support_path, test_path = Path(e['support_manifest']), Path(e['manifest'])
        assert sha(support_path) == e['support_manifest_sha256']
        assert sha(test_path) == e['sha256']
        support = json.loads(support_path.read_text())
        test = json.loads(test_path.read_text())
        rows = support['support_rows']
        labels = np.array([r['labels'] for r in rows], dtype=int)
        assert labels.shape == (e['support_count'], len(e['class_names']))
        assert np.isin(labels, [0, 1]).all()
        images = [r['image'] for r in rows]
        test_images = [r['image'] for r in test['rows']]
        assert not set(images) & set(test_images)
        assert len(test_images) == e['count'] and len(set(test_images)) == len(test_images)
        groups = [str(r.get('patient_id') or r.get('group') or r.get('byte_sha256') or r['image']) for r in rows]
        refs = {v['model']: {'path':v['result'], 'sha256':v['sha256']} for v in e['comparison_references']}
        bio = next(r for r in aggregate['records'] if r['model']=='BiomedCLIP' and r['dataset']==key)
        refs['BiomedCLIP'] = {'path':bio['result_file'], 'sha256':sha(bio['result_file'])}
        for ref in refs.values():
            assert sha(ref['path']) == ref['sha256']
        spatial = None
        if key == 'chestxdet10':
            s = catalog['chestxdet10_gt']
            spatial = {'path':s['manifest'], 'sha256':s['sha256'], 'count':s['count'],
                       'reference':s['comparison_references'][0]}
            assert sha(spatial['path']) == spatial['sha256']
        record = {
            'version':'native-query-fair-v1', 'status':'frozen',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'dataset':key, 'class_names':e['class_names'], 'eta':0.3, 'crop_fraction':0.8,
            'support_manifest':{'path':str(support_path),'sha256':sha(support_path)},
            'test_manifest':{'path':str(test_path),'sha256':sha(test_path)},
            'support_rows':rows, 'support_groups':groups,
            'test_images':test_images, 'support_count':len(rows), 'test_count':len(test_images),
            'spatial_manifest':spatial, 'references':refs,
            'folds':make_folds(labels, groups),
            'selection':{'data':'support_only', 'folds':5, 'seed':42,
                         'metric':'macro_trapezoidal_auprc', 'tie':'first_predeclared_candidate',
                         'test_labels_allowed_in_fit':False},
            'fixed_grids':{'score_alpha':[0,.03,.1,.3,1,3,10], 'tip_beta':[1,2,5,10],
                           'gda_alpha':[0,.0001,.001,.01,.1,1],
                           'simnl_epochs':[0,10,20,40],
                           'simnl_fixed':{'alpha':5,'beta':1,'lambda':.75,'lr':.0001}},
            'limitations':['Previously exposed development datasets; not untouched external validation.',
                'Support-only cross-validation adds no external labels but only one support draw is evaluated.',
                'RSNA patient IDs from source; ChestXDet10 has image/hash groups only, no verified patient IDs.',
                'No requirement that an adaptation outperform the original; all fixed candidates reported.'],
        }
        path = output/(key+'.json')
        write(path, record)
        files[key] = {'path':str(path),'sha256':sha(path),'support_count':len(rows),'test_count':len(test_images)}
    write(output/'catalog.json', {'status':'frozen','datasets':files, 'source_catalog_sha256':sha(catalog_path)})
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args().output)
