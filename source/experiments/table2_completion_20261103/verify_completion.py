#!/usr/bin/env python3
"""Independently audit all seven completed Table 2 additions using CPU only.

No evaluator/model modules are imported. This script reads existing predictions,
recomputes sklearn ranking metrics, validates frozen identities and file hashes,
and independently reconstructs support-only query updates with NumPy. A verified
document is written only after all seven full evaluations pass every check.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
EXPECTED = {
    ('AlphaRAD', 'openi'): (90, 7294),
    ('AlphaRAD', 'siim_radzero_holdout'): (10, 1694),
    ('RadZero', 'chexpert'): (60, 596),
    ('RadZero', 'shenzhenxray'): (10, 652),
    ('RadZero', 'montgomery'): (10, 128),
    ('RadZero', 'covidkaggle'): (10, 1381),
    ('RadZero', 'siim_pneumothorax'): (10, 1704),
}
DEFAULT_RESULTS = [
    HERE / 'alpharad' / ds / 'result.json' for ds in ('openi', 'siim_radzero_holdout')
] + [
    HERE / 'radzero' / group / ds / 'result.json'
    for group, datasets in (
        ('full_gpu6', ('chexpert', 'shenzhenxray', 'montgomery')),
        ('full_gpu7', ('siim_pneumothorax', 'covidkaggle')),
    ) for ds in datasets
]
ALPHA_WEIGHT_SHA = '97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9'
CENTER_VIEW = {'operation': 'crop_resize', 'bounds_xyxy': [52, 52, 466, 466],
               'input_hw': [518, 518], 'output_hw': [518, 518],
               'resize_mode': 'bilinear', 'align_corners': False, 'requested_crop_fraction': .8}
VERIFIED_FILES = {}
HASH_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def verify_hash(path, expected):
    path = Path(path).resolve()
    actual = sha(path)
    require(actual == expected, f'Changed file: {path}; expected {expected}, got {actual}')
    VERIFIED_FILES[str(path)] = actual
    return actual


def verify_hashes(items):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: verify_hash(*item), items.items()))


def finite_array(path, shape=None):
    value = np.load(path, allow_pickle=False)
    require(np.isfinite(value).all(), f'Nonfinite array: {path}')
    if shape is not None:
        require(value.shape == shape, f'Unexpected array shape {path}: {value.shape} != {shape}')
    return value


def metric_values(labels, scores, classes):
    require(labels.shape == scores.shape, 'Score/label dimensions differ')
    per_class = {}
    for column, name in enumerate(classes):
        y, score = labels[:, column], scores[:, column]
        positives = int(y.sum())
        require(0 < positives < len(y), f'Full test class {name} lacks positive/negative outcomes')
        precision, recall, _ = precision_recall_curve(y, score)
        per_class[name] = {
            'positives': positives, 'negatives': len(y) - positives,
            'auroc': float(roc_auc_score(y, score)),
            'auprc': float(auc(recall, precision)),
            'average_precision': float(average_precision_score(y, score)),
        }
    return {
        'per_class': per_class,
        **{'macro_' + metric: float(np.mean([v[metric] for v in per_class.values()]))
           for metric in ('auroc', 'auprc', 'average_precision')},
    }


def compare_metrics(actual, recorded, context):
    for key in ('macro_auroc', 'macro_auprc', 'macro_average_precision'):
        require(key in recorded and recorded[key] is not None, f'Missing {context}/{key}')
        require(abs(actual[key] - recorded[key]) <= 1e-12,
                f'Metric mismatch {context}/{key}: {actual[key]} vs {recorded[key]}')
    for name, values in actual['per_class'].items():
        reference = recorded['per_class'][name]
        for metric in ('auroc', 'auprc', 'average_precision', 'positives', 'negatives'):
            if metric in reference:
                require(abs(values[metric] - reference[metric]) <= 1e-12,
                        f'Per-class mismatch {context}/{name}/{metric}')


def row_identity(rows):
    return [(r['image'], r['labels']) for r in rows]


def frozen_inputs(dataset):
    catalog = read(ROOT / 'baselines/evaluation/matched_phase6_20261022/catalog.json')
    entry = catalog['datasets'][dataset]
    test_path = Path(entry['manifest'])
    verify_hash(test_path, entry['sha256'])
    test = read(test_path)
    verify_hash(test['source']['csv'], test['source']['sha256'])
    source = ROOT / 'baselines/evaluation/query_transfer_20261022_v1' / (dataset + '.json')
    support = read(source)
    verify_hash(test_path, support['test_manifest_sha256'])
    verify_hashes(support['support_sources'])
    require(support['class_names'] == test['class_names'] == entry['class_names'], 'Class order mismatch')
    require(test['count'] == len(test['rows']) == support['count'], 'Frozen count mismatch')
    require(len(support['support_rows']) == support['support_count'], 'Support count mismatch')
    verify_hashes({r['image']: r['byte_sha256'] for r in support['support_rows']})
    return test, support, source


def check_disjoint(support_rows, test_rows, native=False):
    key = 'native_image' if native else 'image'
    support_paths = {str(Path(r[key]).resolve()) for r in support_rows}
    test_paths = {str(Path(r[key]).resolve()) for r in test_rows}
    require(len(support_paths) == len(support_rows) and len(test_paths) == len(test_rows), 'Duplicate image rows')
    require(not support_paths & test_paths, 'Support/test image overlap')
    linkage = {}
    for field in ('patient_id', 'study_id', 'group'):
        a = {r[field] for r in support_rows if r.get(field)}
        b = {r[field] for r in test_rows if r.get(field)}
        require(not a & b, f'Support/test identifier overlap: {field}')
        linkage[field] = {'support_known': len(a), 'test_known': len(b), 'overlap': 0}
    return linkage


def query_audit(features, labels, queries):
    require(features.shape[:2] == labels.shape and features.shape[1:] == queries.shape[1:], 'Query shape mismatch')
    require(queries.shape[0] == 2 and np.isfinite(features).all() and np.isfinite(queries).all(), 'Invalid queries/features')
    require(np.isin(labels, [0, 1]).all(), 'Nonbinary support labels')
    positive_counts = labels.sum(0)
    require(np.all((positive_counts > 0) & (positive_counts < len(labels))), 'Missing support outcome')
    x, y, q0 = features.astype(np.float64), labels.astype(np.float64), queries[0].astype(np.float64)
    norms = np.linalg.norm(x, axis=-1)
    np.testing.assert_allclose(norms, 1., atol=3e-6, rtol=0.)
    difference = (x * y[..., None]).sum(0) / y.sum(0)[:, None] - (x * (1-y)[..., None]).sum(0) / (1-y).sum(0)[:, None]
    direction = difference / np.maximum(np.linalg.norm(difference, axis=-1, keepdims=True), 1e-12)
    combined = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12) + .3 * direction
    expected = combined / np.maximum(np.linalg.norm(combined, axis=-1, keepdims=True), 1e-12)
    np.testing.assert_allclose(queries[1], expected, atol=3e-6, rtol=0.)
    return {'passed': True, 'method': 'independent float64 NumPy support-only direction and eta=0.3 update',
            'query_max_abs_error': float(np.max(np.abs(queries[1] - expected))),
            'positive_counts': positive_counts.tolist(), 'negative_counts': (len(labels)-positive_counts).tolist(),
            'feature_norm_max_abs_error': float(np.max(np.abs(norms-1)))}


def alpha_audit(directory, result, test, support):
    audit = read(directory / 'data_audit.json')
    verify_hash(directory / 'data_audit.json', result['data_audit_sha256'])
    verify_hash(directory / 'provenance.json', result['provenance_sha256'])
    provenance = read(directory / 'provenance.json')
    verify_hashes(provenance['source_sha256'])
    snapshot_path = HERE / 'alpharad/frozen_sources.json'
    snapshot = read(snapshot_path)
    require(str(HERE / 'alpharad/evaluate_missing.py') in snapshot, 'AlphaRAD driver absent from frozen source snapshot')
    verify_hashes(snapshot)
    require(provenance['checkpoint_sha256'] == result['checkpoint_sha256'] == ALPHA_WEIGHT_SHA, 'AlphaRAD checkpoint identity mismatch')
    verify_hash(provenance['checkpoint'], ALPHA_WEIGHT_SHA)
    require(provenance['eta'] == .3 and provenance['weight_updates'] is False and provenance['selection_performed'] is False and provenance['test_labels_used_for_query_fit'] is False, 'AlphaRAD protocol changed')
    require(set(provenance['views']) == {'identity', 'center80'} and provenance['views']['center80'] == CENTER_VIEW, 'AlphaRAD view geometry changed')
    verify_hash(audit['source_test_manifest'], audit['source_test_manifest_sha256'])
    verify_hash(audit['source_test_csv'], audit['source_test_csv_sha256'])
    verify_hash(audit['source_support_csv'], audit['source_support_csv_sha256'])
    verify_hashes({directory / k: v for k, v in audit['derived_file_sha256'].items()})
    actual_support = read(directory / 'support_manifest.json')['rows']
    actual_test = read(directory / 'test_manifest.json')['rows']
    require(row_identity(actual_support) == row_identity(support['support_rows']), 'AlphaRAD support rows differ from frozen rows')
    require(row_identity(actual_test) == row_identity(test['rows']), 'AlphaRAD test rows differ from frozen rows')
    hashes = read(directory / 'image_hashes.json')
    require(set(hashes) == {r['image'] for r in actual_support + actual_test}, 'Incomplete AlphaRAD image hashes')
    verify_hashes(hashes)
    require(not {hashes[r['image']] for r in actual_support} & {hashes[r['image']] for r in actual_test}, 'AlphaRAD exact-byte overlap')
    adapter = read(directory / 'query_adapter.json')
    verify_hash(directory / 'query_adapter.npz', adapter['query_sha256'])
    require(adapter['phase6_queries_bitwise_equal'] is True and result['phase6_query_fit_bitwise_equal'] is True, 'AlphaRAD phase6 query check missing')
    require(adapter['eta'] == .3 and adapter['class_names'] == test['class_names'] and adapter['support_views'] == ['identity'] and adapter['test_labels_used_for_query_fit'] is False, 'AlphaRAD adapter settings changed')
    labels = np.asarray([r['labels'] for r in actual_support], dtype=np.uint8)
    features = finite_array(directory / 'support_class_local.npy')
    with np.load(directory / 'query_adapter.npz', allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle['etas'], [0., .3])
        np.testing.assert_array_equal(bundle['positive_counts'], labels.sum(0))
        qcheck = query_audit(features, labels, bundle['queries'])
    historical = read(directory / 'historical_check.json')
    require(historical == result['historical_check'] and historical['passed'] is True and historical['images'] == 8, 'AlphaRAD historical regression check missing')
    verify_hashes(historical['reference_sha256'])
    require(set(historical['probability_max_abs_errors']) == set(BRANCHES), 'Incomplete AlphaRAD historical regression branches')
    require(all(0 <= e <= 1.2e-5 for e in historical['probability_max_abs_errors'].values()), 'AlphaRAD historical regression error')
    native = read(directory / 'native_helper_check.json')
    require(native == result['native_helper_check'] and native['passed'] is True, 'AlphaRAD native helper check missing')
    whole_test_regression = None
    if result['dataset'] == 'siim_radzero_holdout':
        prior_dir = Path(historical['historical_run'])
        prior_paths = read(prior_dir / 'image_paths.json')
        require(len(prior_paths) == len(set(prior_paths)) == 1704, 'Historical SIIM path list changed')
        lookup = {path: index for index, path in enumerate(prior_paths)}
        indices = [lookup[row['image']] for row in actual_test]
        prior_labels = finite_array(prior_dir / 'labels.npy')[indices]
        np.testing.assert_array_equal(prior_labels, np.asarray([row['labels'] for row in actual_test]))
        errors = {}
        for branch in ('original', 'tta_center_original'):
            old_path = prior_dir / 'candidates' / branch / 'probabilities.npy'
            expected = finite_array(old_path)[indices]
            current = finite_array(directory / 'branches' / branch / 'probabilities.npy')
            np.testing.assert_allclose(current, expected, atol=2e-6, rtol=1e-5)
            errors[branch] = float(np.max(np.abs(current-expected)))
        whole_test_regression = {'passed': True, 'images': 1694, 'branches': errors,
                                 'reason': 'Original and V are support-independent; historical 1704-image predictions subsetted by exact path.'}
    return {'query': qcheck, 'native_regression': native, 'historical_regression': historical,
            'linkage': check_disjoint(actual_support, actual_test), 'checkpoint': provenance['checkpoint'],
            'source_code_files': len(provenance['source_sha256']),
            'full_test_original_and_v_regression': whole_test_regression,
            'frozen_source_snapshot': {'path': str(snapshot_path), 'sha256': sha(snapshot_path), 'verified_sources': len(snapshot)},
            'support_features': {'path': str(directory / 'support_class_local.npy'), 'sha256': sha(directory / 'support_class_local.npy')}}


def rad_audit(directory, result, test, support):
    verify_hashes({directory / k: v for k, v in result['artifact_sha256'].items()})
    config = read(directory / 'run_config.json')
    require(config == result['config'], 'RadZero config copy mismatch')
    require(config['is_smoke'] is False and config['eta'] == .3 and config['views'] == ['identity', 'center80'] and config['support_views'] == ['identity'], 'RadZero protocol changed')
    require(config['view_specs'] == CENTER_VIEW, 'RadZero view geometry changed')
    require(config['trainable_base_parameters'] == 0 and config['test_labels_used_for_query_fit'] is False and config['tuning_or_selection'] is False, 'RadZero adaptation policy changed')
    verify_hashes(config['source_sha256'])
    verify_hashes(config['source_manifests'])
    model_dir = next(Path(p).parent for p in config['source_sha256'] if p.endswith('/preprocessor_config.json'))
    download = read(model_dir / 'download_manifest.json')
    expected_weight = next(r['hash'] for r in download['files'] if r['path'] == 'model.safetensors')
    require(config['checkpoint'] == {'repo': download['repo'], 'revision': download['revision'], 'weight_sha256': expected_weight}, 'RadZero checkpoint provenance mismatch')
    verify_hash(model_dir / 'model.safetensors', expected_weight)
    inputs = read(directory / 'input_manifest.json')
    require(inputs['class_names'] == test['class_names'], 'RadZero class order changed')
    actual_support, actual_test = inputs['support_rows'], inputs['test_rows']
    require(row_identity(actual_support) == row_identity(support['support_rows']), 'RadZero support rows changed')
    require(row_identity(actual_test) == row_identity(test['rows']), 'RadZero test rows changed')
    verify_hashes(inputs['source_sha256'])
    verify_hashes({r['native_image']: r['native_byte_sha256'] for r in actual_support + actual_test})
    require(not {r['native_byte_sha256'] for r in actual_support} & {r['native_byte_sha256'] for r in actual_test}, 'RadZero exact-byte overlap')
    if result['dataset'] == 'siim_pneumothorax':
        for row in actual_support + actual_test:
            require(Path(row['native_image']).stem == Path(row['image']).stem, 'SIIM DICOM UID remapping mismatch')
        require(not {Path(r['native_image']).stem for r in actual_support} & {Path(r['native_image']).stem for r in actual_test}, 'SIIM support/test UID overlap')
        support_csv = next(Path(p) for p in support['support_sources'] if p.endswith('_support.csv'))
        source_supports = {r['image']: r for r in csv.DictReader(support_csv.open())}
        for row in actual_support:
            require(Path(row['native_image']).resolve() == Path(source_supports[row['image']]['source_image']).resolve(), 'SIIM support DICOM source mismatch')
    else:
        require(all(r['native_image'] == r['image'] for r in actual_support + actual_test), 'Unexpected native path remapping')
    synonyms = {'Consolidation': 'Pulmonary Consolidation', 'Edema': 'Pulmonary Edema', 'Fracture': 'Bone Fracture'}
    expected_prompts = ['There is ' + synonyms.get(c, c) for c in test['class_names']]
    require(config['prompts'] == inputs['prompts'] == expected_prompts, 'RadZero fixed prompt mismatch')
    with np.load(directory / 'support_features.npz', allow_pickle=False) as features, np.load(directory / 'queries.npz', allow_pickle=False) as queries:
        labels = np.asarray([r['labels'] for r in actual_support], dtype=np.int8)
        np.testing.assert_array_equal(features['labels'], labels)
        require(queries['names'].tolist() == ['original', 'query_refined'], 'RadZero query branch order changed')
        qcheck = query_audit(features['features'], labels, queries['queries'])
    checks = read(directory / 'verification.json')
    require(checks['phase6_query_bitwise_equal'] is True and len(checks['official_api']) >= 1, 'RadZero native/query check missing')
    require(checks['pooled_reconstruction_max_error'] <= 1e-4, 'RadZero pooled reconstruction mismatch')
    for entry in checks['official_api']:
        require(entry['classes'] == len(test['class_names']) and entry['max_logit_error'] <= 1e-3 and entry['max_map_error'] <= 1e-3, 'RadZero native API mismatch')
    historical = read(config['historical_equivalence'])
    for branch in BRANCHES:
        check = historical[branch]
        require(check['passed'] is True and check['rows'] == 16 and 0 <= check['max_abs_error'] <= 2e-5, 'RadZero historical regression mismatch')
        verify_hash(check['historical_predictions'], check['historical_sha256'])
    require(historical['historical_query_fit'] == {'passed': True, 'bitwise_equal': True}, 'RadZero historical query-fit mismatch')
    return {'query': qcheck, 'native_regression': checks, 'historical_regression': historical,
            'linkage': check_disjoint(actual_support, actual_test),
            'native_linkage': check_disjoint(actual_support, actual_test, native=True),
            'checkpoint': str(model_dir / 'model.safetensors'), 'source_code_files': len(config['source_sha256'])}


def verify_result(path, result):
    model, dataset = result['model'], result['dataset']
    directory = path.parent
    print(f'Verifying {model} / {dataset}: identities, code, weights and images', flush=True)
    test, support, support_source = frozen_inputs(dataset)
    expected_support, expected_test = EXPECTED[model, dataset]
    require(result['rows'] == test['count'] == expected_test and result['support_count'] == support['support_count'] == expected_support, 'Expected sample count mismatch')
    require(result['class_names'] == test['class_names'], 'Result class order mismatch')
    require(result['no_network_weight_updates'] is True, 'Network weight updates declared')
    if model == 'AlphaRAD':
        verify_hash(directory / 'labels.npy', result['labels_sha256'])
        verify_hash(directory / 'image_paths.json', result['image_paths_sha256'])
        details = alpha_audit(directory, result, test, support)
    else:
        details = rad_audit(directory, result, test, support)
    classes = test['class_names']
    labels = finite_array(directory / 'labels.npy', (expected_test, len(classes)))
    np.testing.assert_array_equal(labels, np.asarray([r['labels'] for r in test['rows']]))
    require(read(directory / 'image_paths.json') == [r['image'] for r in test['rows']], 'Saved test path order mismatch')
    require(set(result['branches']) == set(BRANCHES), 'Incomplete or extra branches')
    branch_audits = {}
    for branch in BRANCHES:
        entry = result['branches'][branch]
        if model == 'AlphaRAD':
            metrics = entry['metrics']
            files = {kind: Path(entry['files'][kind]) for kind in ('logits', 'probabilities')}
            for kind, file in files.items():
                require(file.resolve() == (directory / 'branches' / branch / (kind + '.npy')).resolve(), 'AlphaRAD prediction path mismatch')
                verify_hash(file, entry['sha256'][kind])
        else:
            metrics = entry
            branch_dir = Path(entry['directory'])
            require(branch_dir.resolve() == (directory / branch).resolve(), 'RadZero prediction path mismatch')
            files = {kind: branch_dir / (kind + '.npy') for kind in ('logits', 'probabilities')}
            for kind, file in files.items():
                verify_hash(file, entry[kind + '_sha256'])
            require(read(branch_dir / 'metrics.json') == entry, 'RadZero saved metric file differs')
        logits, probabilities = (finite_array(files[k], labels.shape) for k in ('logits', 'probabilities'))
        require(np.all((probabilities >= 0) & (probabilities <= 1)), 'Invalid probabilities')
        expected_probabilities = 1. / (1. + np.exp(-logits.astype(np.float64)))
        np.testing.assert_allclose(probabilities, expected_probabilities,
                                   atol=1e-7 if model == 'AlphaRAD' else 2e-15, rtol=0.)
        actual = metric_values(labels, probabilities if model == 'AlphaRAD' else logits, classes)
        compare_metrics(actual, metrics, f'{model}/{dataset}/{branch}')
        if model == 'RadZero':
            probability_metrics = metric_values(labels, probabilities, classes)
            for key in ('macro_auroc', 'macro_auprc', 'macro_average_precision'):
                require(actual[key] == probability_metrics[key], f'Sigmoid changes RadZero ranking: {dataset}/{branch}/{key}')
        branch_audits[branch] = {'metrics': actual, 'score_domain': 'probabilities' if model == 'AlphaRAD' else 'native_logits',
                                'files': {k: {'path': str(v), 'sha256': sha(v)} for k, v in files.items()},
                                'sigmoid_max_abs_error': float(np.abs(probabilities-expected_probabilities).max())}
    details.update({'passed': True, 'rows': expected_test, 'support_count': expected_support, 'class_names': classes,
                    'test_positive_counts': labels.sum(0).tolist(), 'support_source': str(support_source),
                    'support_source_sha256': sha(support_source), 'branches': branch_audits,
                    'regression_validation': 'Saved native/historical checks and their reference hashes validated; no GPU rerun by this CPU verifier.'})
    print(f'PASS {model} / {dataset}: all four AUROC/AUPRC/AP metrics reproduced', flush=True)
    return {'model': model, 'dataset': dataset, 'result': str(path), 'sha256': sha(path), 'prediction_audit': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+', help='Exactly seven full result.json paths; defaults to agreed final run directories.')
    parser.add_argument('--output', type=Path, default=HERE / 'VERIFIED_COMPLETION.json')
    parser.add_argument('--audit-output', type=Path, default=HERE / 'verification_file_hashes.json')
    args = parser.parse_args()
    paths = [p.resolve() for p in (args.results or DEFAULT_RESULTS)]
    require(len(paths) == len(EXPECTED) and len(set(paths)) == len(paths), 'Require exactly seven distinct result paths')
    missing = [str(p) for p in paths if not p.is_file()]
    require(not missing, 'Missing complete result files: ' + ', '.join(missing))
    results = [read(p) for p in paths]
    for path, result in zip(paths, results):
        require(result.get('status') == 'complete' and result.get('is_smoke') is False and result.get('task') == 'classification', f'Incomplete/smoke/nonclassification result rejected: {path}')
    pairs = [(r['model'], r['dataset']) for r in results]
    require(len(set(pairs)) == len(EXPECTED) and set(pairs) == set(EXPECTED), f'Missing, duplicate or unexpected model/dataset pairs: {pairs}')
    records = [verify_result(p, r) for p, r in zip(paths, results)]
    verifier = Path(__file__).resolve()
    audit = {'status': 'verified', 'verifier': str(verifier), 'verifier_sha256': sha(verifier),
             'file_count': len(VERIFIED_FILES), 'verified_file_sha256': dict(sorted(VERIFIED_FILES.items()))}
    args.audit_output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    document = {'status': 'verified', 'prediction_metrics_verified': True,
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'verifier': str(verifier), 'verifier_sha256': sha(verifier),
                'numpy_version': np.__version__, 'cpu_only': True,
                'metric_definition': 'Per-class AUROC, trapezoidal auc(recall,precision), and average_precision_score; equal class macro averages.',
                'verified_file_hash_audit': str(args.audit_output.resolve()), 'verified_file_hash_audit_sha256': sha(args.audit_output),
                'records': records}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(f'Verified all {len(records)} pairs, {len(records)*len(BRANCHES)} branch metric sets and {len(VERIFIED_FILES)} hashed files: {args.output}', flush=True)


if __name__ == '__main__':
    main()
