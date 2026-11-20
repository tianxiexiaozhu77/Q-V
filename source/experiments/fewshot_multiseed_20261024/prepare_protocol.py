#!/usr/bin/env python3
"""Freeze the eleven original comparators on the repaired five-support protocol.

Supports, candidate pool, fixed tests and resampling groups are inherited exactly.
The original support-only fold builder and every algorithm/grid remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/fewshot_native_query'
STABILITY = ROOT / 'experiments/rsna_chest_stability_20261024'
SOURCE = STABILITY / 'protocol_v1/catalog.json'
RUN = STABILITY / 'runs/full_20261024_v1'
SEEDS = (42, 43, 44, 45, 46)
METHODS = ('original', 'positive_query', 'query', 'view', 'query_view',
           'prototype_score_only', 'prototype_score_mix_oof', 'tip_adapter_ovr_oof',
           'gda_ovr_ks_oof', 'class_local_difference_score_oof', 'simnl_feature_ovr')
VERSION = 'native-query-eleven-methods-repaired-five-support-seeds-v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def checked(spec):
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Frozen source changed: ' + spec['path'])
    return json.loads(Path(spec['path']).read_text())


def frozen(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError('Refusing to alter frozen protocol: ' + str(path))
    if not path.exists():
        path.write_bytes(encoded)
    return ref(path)


def fold_audit(rows, groups, folds):
    labels, groups = np.asarray([r['labels'] for r in rows]), np.asarray(groups)
    coverage = []
    audit = []
    for i, fold in enumerate(folds):
        tr, va = fold['train'], fold['validation']
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('Support group crosses CV fold')
        coverage.extend(va)
        positive = labels[tr].sum(0)
        missing = np.flatnonzero((positive == 0) | (positive == len(tr))).tolist()
        audit.append({'fold': i, 'train_count': len(tr), 'validation_count': len(va),
                      'train_positive_counts': positive.tolist(),
                      'train_negative_counts': (len(tr)-positive).tolist(),
                      'validation_positive_counts': labels[va].sum(0).tolist(),
                      'validation_negative_counts': (len(va)-labels[va].sum(0)).tolist(),
                      'missing_training_binary_class_indices': missing,
                      'train_validation_group_overlap': 0})
    if sorted(coverage) != list(range(len(rows))):
        raise ValueError('OOF validation does not cover support exactly once')
    return audit


def prepare(output):
    catalog = json.loads(SOURCE.read_text())
    if catalog['status'] != 'frozen' or catalog['support_seeds'] != list(SEEDS):
        raise ValueError('Expected repaired frozen five-seed source')
    spec = importlib.util.spec_from_file_location('_original_fewshot_prepare', OLD/'prepare_protocol.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    sources = {name: ref(OLD/name) for name in ('prepare_protocol.py', 'methods.py', 'simnl_adapter.py')}
    sources['spatial_hits.py'] = ref(STABILITY/'infer_multiseed.py')
    datasets, audits = {}, {}
    for dataset, source_ref in catalog['datasets'].items():
        source = checked(source_ref)
        old = checked(source['historical_protocol'])
        pool = checked(source['candidate_pool'])
        test_groups = set(source['test_groups'])
        test_images = [r['image'] for r in source['test_rows']]
        candidate_groups = {r['group_id'] for r in pool['candidate_rows'] if r.get('group_id')}
        candidate_overlap = len(candidate_groups & test_groups) if candidate_groups else None
        if candidate_overlap:
            raise ValueError('Candidate pool overlaps test patients')
        record = copy.deepcopy(source)
        record.update(version=VERSION, methods=list(METHODS), branches=list(METHODS),
                      source_protocol=source_ref, source_catalog=ref(SOURCE),
                      algorithm_sources=sources, fixed_grids=old['fixed_grids'])
        seed_refs, seed_audits = {}, {}
        for seed in SEEDS:
            entry = record['seeds'][str(seed)]
            seed_old = checked(entry['native_protocol'])
            rows, groups = entry['support_rows'], entry['support_groups']
            labels = np.asarray([r['labels'] for r in rows], dtype=int)
            if len(rows) != (10 if dataset == 'rsna_pneumonia' else 50):
                raise ValueError('Inherited support budget changed')
            if set(groups) & test_groups or set(r['image'] for r in rows) & set(test_images):
                raise ValueError('Inherited support overlaps fixed test')
            if len(set(groups)) != len(groups):
                raise ValueError('Support groups must be independent')
            # Preserve original function, fixed search budget, scoring and tie rule.
            folds = original.make_folds(labels, groups, seed=42, nfolds=5)
            fa = fold_audit(rows, groups, folds)
            degenerate = any(r['missing_training_binary_class_indices'] for r in fa)
            if degenerate:
                raise ValueError('Original fold builder returned a degenerate partition')
            references = {}
            for model in catalog['models']:
                folder = RUN / f'{model}_{dataset}' / f'seed_{seed}'
                references[model] = {name: ref(folder/name) for name in ('result.json', 'predictions.npz')}
                if (folder/'spatial_result.json').exists():
                    references[model]['spatial_result.json'] = ref(folder/'spatial_result.json')
            audit = {'support_count': len(rows), 'support_group_count': len(set(groups)),
                     'positive_counts': labels.sum(0).tolist(), 'negative_counts': (len(rows)-labels.sum(0)).tolist(),
                     'support_test_group_overlap': 0, 'support_test_image_overlap': 0,
                     'all_training_folds_have_both_outcomes': not degenerate, 'folds': fa,
                     'fallback_triggered': False,
                     'fallback_rule': 'Original fold builder fails if no valid 5-fold partition in 2000 draws. Original score methods mark nonzero candidates invalid and choose alpha=0 if supplied folds lack an outcome; SimNL rejects invalid folds. No fold/seed is replaced using test outcomes.'}
            p = copy.deepcopy(seed_old)
            p.update(version=VERSION, support_seed=seed, folds=folds, fixed_grids=old['fixed_grids'],
                     methods=list(METHODS), test_groups=source['test_groups'],
                     test_patient_ids=[r.get('patient_id', '') for r in source['test_rows']],
                     source_protocol=source_ref, source_catalog=ref(SOURCE),
                     source_native_protocol=entry['native_protocol'],
                     algorithm_sources=sources, stability_references=references,
                     selection={'data':'support_only', 'folds':5, 'seed':42,
                                'support_seed':seed, 'search_candidates':2000,
                                'metric':'macro_trapezoidal_auprc',
                                'tie':'original method tie rules: first predeclared score candidate within 1e-12; SimNL smallest epoch at exact ties',
                                'test_labels_allowed_in_fit':False,
                                'same_folds_across_models_and_methods':True},
                     support_cv_audit=audit, limitations=source['limitations'] + [
                         'Five prespecified supports and every original method are reported without choosing a winning seed.',
                         'The 10 RSNA supports have 5 negative and 5 positive examples; CV is intrinsically small.',
                         'ChestXDet10 is multilabel; grouped folds preserve both outcomes in training, while validation folds can lack a class. Selection uses pooled OOF predictions.'])
            destination = output/dataset/f'seed_{seed}'/'native_protocol.json'
            new_ref = frozen(destination, p)
            entry['source_native_protocol'] = entry['native_protocol']
            entry['native_protocol'] = new_ref
            entry['folds'] = folds
            entry['support_cv_audit'] = audit
            seed_refs[str(seed)], seed_audits[str(seed)] = new_ref, audit
        record['candidate_pool_patient_overlap_audit'] = {
            'candidate_count':len(pool['candidate_rows']), 'candidate_group_count':len(candidate_groups) or None,
            'candidate_test_group_overlap':candidate_overlap,
            'patient_identity_available':dataset == 'rsna_pneumonia',
            'note':'RSNA candidate-pool patient isolation checked in full; ChestXDet10 candidate images are metadata-only and patient identity is unavailable. Inherited selected-support byte/pixel audit is preserved.'}
        record['limitations'] = p['limitations']
        new_ref = frozen(output/(dataset+'.json'), record)
        datasets[dataset] = dict(new_ref, test_count=source['test_count'], test_group_count=source['test_group_count'],
                                support_count_per_seed=len(rows), support_union_count=source['support_union_count'],
                                resampling_unit=source['resampling_unit'], class_names=source['class_names'], seeds=seed_refs)
        audits[dataset] = dict(source_protocol=source_ref, candidate_pool=record['candidate_pool_patient_overlap_audit'],
                              inherited_source_audit=source['source_audit'], seeds=seed_audits)
    audit_ref = frozen(output/'PROTOCOL_AUDIT.json', {'status':'passed', 'source_catalog':ref(SOURCE), 'datasets':audits})
    result = dict(version=VERSION, status='frozen', support_seeds=list(SEEDS), models=catalog['models'],
                  methods=list(METHODS), datasets=datasets, eta=.3, crop_fraction=.8,
                  source_catalog=ref(SOURCE), algorithm_sources=sources, audit=audit_ref,
                  bootstrap_replicates=2000, bootstrap_seed=20261024,
                  protocol_preparation_script=ref(__file__))
    frozen(output/'catalog.json', result)
    print(json.dumps({'catalog':str(output/'catalog.json'), 'sha256':sha(output/'catalog.json'), 'status':'frozen'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'protocol_v1')
    prepare(parser.parse_args().output.resolve())
