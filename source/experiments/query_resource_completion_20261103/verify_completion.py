#!/usr/bin/env python3
"""CPU-only independent audit of six fixed query-resource extensions.

This verifier imports no model code and does not run inference. It checks
unrounded metrics, per-view probability fusion, frozen support/test identities,
support-derived norm-preserving query updates, unchanged original contexts and
checkpoint/source provenance. Nothing is promoted until all six runs pass.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HELPER = ROOT / 'experiments/table2_completion_20261103/verify_completion.py'
spec = importlib.util.spec_from_file_location('previous_cpu_verification', HELPER)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
read, sha, require = common.read, common.sha, common.require
verify_hash, verify_hashes = common.verify_hash, common.verify_hashes
BRANCHES = common.BRANCHES
DESCRIPTIONS = HERE / 'query_descriptions.json'
DESCRIPTION_SHA = 'aa7e8ce55477f6b8aee09f11448fdb08688b0a89a0ef9e2652e43c55aa4fd778'
DATASETS = {'openi': (90, 7294), 'shenzhenxray': (10, 652), 'montgomery': (10, 128)}
EXPECTED = {(model, dataset) for model in ('MedKLIP', 'MAVL') for dataset in DATASETS}
DEFAULT_RESULTS = [HERE / directory / dataset / 'result.json'
                   for directory in ('medklip', 'mavl') for dataset in DATASETS]


def array(bundle, key):
    value = np.asarray(bundle[key])
    require(np.isfinite(value).all(), f'Nonfinite array: {key}')
    return value


def normalize(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def verify_support_queries(model, features, labels, original, refined, fit):
    """Recompute with float64 NumPy, independently of torch fitting helpers."""
    require(features.shape == (len(labels), 256), f'Unexpected {model} support feature shape')
    expected_shape = (labels.shape[1], 256) if model == 'MedKLIP' else (labels.shape[1], 9, 256)
    require(original.shape == refined.shape == expected_shape, f'Unexpected {model} query shape')
    require(np.isfinite(features).all() and np.isfinite(original).all() and np.isfinite(refined).all(), 'Nonfinite support/query')
    require(np.isin(labels, [0, 1]).all(), 'Nonbinary support labels')
    positives = labels.sum(0)
    negatives = len(labels) - positives
    require(np.all(positives > 0) and np.all(negatives > 0), 'Missing support class outcome')
    x, y, q = normalize(features.astype(np.float64)), labels.astype(np.float64), original.astype(np.float64)
    positive_mean = np.einsum('nd,nc->cd', x, y) / positives[:, None]
    negative_mean = np.einsum('nd,nc->cd', x, 1-y) / negatives[:, None]
    differences = positive_mean - negative_mean
    direction = normalize(differences)
    for key, expected in [('positive_means', positive_mean), ('negative_means', negative_mean),
                          ('directions', direction), ('direction', direction),
                          ('positive_counts', positives), ('negative_counts', negatives)]:
        if key in fit:
            np.testing.assert_allclose(np.asarray(fit[key]), expected, atol=4e-6, rtol=0.)
    broadcast = direction if model == 'MedKLIP' else direction[:, None, :]
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    require(np.all(norms > 1e-12), 'Zero original query')
    expected = normalize(normalize(q) + .3 * broadcast) * norms
    expected = np.where(np.linalg.norm(broadcast, axis=-1, keepdims=True) < 1e-12, q, expected)
    np.testing.assert_allclose(refined, expected, atol=5e-6, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(refined, axis=-1), norms[..., 0], atol=5e-6, rtol=1e-6)
    require(fit['eta'] == .3 and fit['preserve_query_norm'] is True, 'Query adaptation settings changed')
    return {'passed': True, 'independent_method': 'Normalize each support image feature; class-positive minus class-negative means; normalize direction; eta=.3 update preserving each original query norm.',
            'positive_counts': positives.tolist(), 'negative_counts': negatives.tolist(),
            'query_max_abs_error': float(np.abs(refined-expected).max()),
            'query_norm_max_abs_error': float(np.abs(np.linalg.norm(refined, axis=-1)-norms[..., 0]).max())}


def verify_predictions(directory, result, test):
    filename = 'predictions.npz' if (directory / 'predictions.npz').is_file() else 'prediction.npz'
    path = directory / filename
    require(filename in result['artifact_sha256'], 'Prediction archive lacks stored SHA256')
    verify_hash(path, result['artifact_sha256'][filename])
    audits = {}
    with np.load(path, allow_pickle=False) as bundle:
        labels = array(bundle, 'labels')
        np.testing.assert_array_equal(labels, np.asarray([r['labels'] for r in test['rows']]))
        require(bundle['image_paths'].tolist() == [r['image'] for r in test['rows']], 'Prediction path/order mismatch')
        require(bundle['class_names'].tolist() == test['class_names'], 'Prediction class order mismatch')
        require(set(result['branches']) == set(BRANCHES), 'Missing or extra reported branches')
        values = {name: array(bundle, name) for name in BRANCHES}
        center_original = array(bundle, 'center_original')
        center_refined = array(bundle, 'center_refined')
        for name, value in {**values, 'center_original': center_original, 'center_refined': center_refined}.items():
            require(value.shape == labels.shape and np.all((value >= 0) & (value <= 1)), f'Invalid probability array: {name}')
        if 'identity_original' in bundle:
            np.testing.assert_array_equal(bundle['identity_original'], values['original'])
            np.testing.assert_array_equal(bundle['identity_refined'], values['query_refined'])
        expected_v = (values['original'] + center_original) * .5
        expected_qv = (values['query_refined'] + center_refined) * .5
        np.testing.assert_array_equal(values['tta_center_original'], expected_v)
        np.testing.assert_array_equal(values['tta_center_refined'], expected_qv)
        for name, value in values.items():
            metrics = common.metric_values(labels, value, test['class_names'])
            common.compare_metrics(metrics, result['branches'][name], f'{result["model"]}/{result["dataset"]}/{name}')
            audits[name] = metrics
    return {'passed': True, 'path': str(path), 'sha256': sha(path),
            'per_view_probability_fusion_bitwise_equal': True, 'branches': audits}


def verify_frozen_inputs(directory, result):
    dataset = result['dataset']
    test, support, source = common.frozen_inputs(dataset)
    support_count, test_count = DATASETS[dataset]
    require(result['rows'] == len(test['rows']) == test_count, 'Wrong full test sample count')
    require(result['support_count'] == len(support['support_rows']) == support_count, 'Wrong support sample count')
    require(result['class_names'] == test['class_names'], 'Result class order mismatch')
    linkage = common.check_disjoint(support['support_rows'], test['rows'])
    old_path = HERE / 'integration/prior/baselines/paper_draft_zh/verified_results.json'
    old_records = read(old_path)['records']
    matches = [r for r in old_records if (r['model'], r['dataset']) == (result['model'], dataset)]
    require(len(matches) == 1 and matches[0]['status'] == 'unsupported', 'Expected immutable previous unsupported record')
    old = matches[0]
    verify_hash(old['source_path'], old['source_sha256'])
    require(read(old['source_path'])['status'] == 'unsupported', 'Historical unsupported source was overwritten')
    return test, support, {'passed': True, 'linkage': linkage,
        'support_manifest': str(source), 'support_manifest_sha256': sha(source),
        'supersedes': {'result': old['source_path'], 'sha256': old['source_sha256'], 'status': 'unsupported', 'reason': old['reason']}}


def verify_run(path, result):
    """Model-specific artifact layouts are kept explicit and reviewed separately."""
    model, dataset, directory = result['model'], result['dataset'], path.parent
    print(f'Verifying {model}/{dataset}', flush=True)
    verify_hashes({directory / name: digest for name, digest in result['artifact_sha256'].items()})
    verify_hashes(result['source_sha256'])
    require(result['frozen_descriptions_sha256'] == DESCRIPTION_SHA, 'Description resources changed')
    test, support, identities = verify_frozen_inputs(directory, result)
    checks = verify_model_artifacts(directory, result, test, support)
    predictions = verify_predictions(directory, result, test)
    print(f'PASS {model}/{dataset}: four complete branches and unchanged frozen identities', flush=True)
    return {'model': model, 'dataset': dataset, 'result': str(path), 'sha256': sha(path),
            'prediction_audit': {'passed': True, 'rows': result['rows'], 'support_count': result['support_count'],
                                 'identities': identities, 'model_artifacts': checks, 'predictions': predictions}}


def verify_model_artifacts(directory, result, test, support):
    model = result['model']
    model_root = ROOT / 'baselines' / model
    descriptions = read(DESCRIPTIONS)['descriptions']
    source = model_root / 'local_setup/adapter.py'
    tree = ast.parse(source.read_text())
    aliases = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ALIASES' for target in node.targets))
    book_path = (model_root / 'PreTrain_MedKLIP/data_file/observation explanation.json' if model == 'MedKLIP'
                 else model_root / 'Zero-shot_classification/observation explanation.json')
    base_names = [key for key in read(book_path) if model != 'MAVL' or key != 'covid19']
    require(len(base_names) == 75, 'Original disease context is not 75 classes')
    keys = [aliases.get(name.lower(), name.lower()) for name in test['class_names']]
    added_names = [name for name, key in zip(test['class_names'], keys) if key not in base_names]
    added_keys = [key for key in keys if key not in base_names]
    require(added_names == (['Calcified Granuloma', 'Granuloma'] if result['dataset'] == 'openi' else ['Tuberculosis']), 'Wrong missing disease resources')
    expected_context = base_names + added_keys
    expected_indices = [expected_context.index(key) for key in keys]
    require(result['test_labels_used_for_query_fit'] is False and result['no_gradient_training'] is True, 'Unexpected training/selection')
    if model == 'MedKLIP':
        require(result['all_four_branches_share_context'] is True and result['query_context_unchanged_during_inference'] is True, 'Context changed across branches')
        require(read(directory / 'manifest.json') == support and read(directory / 'test_manifest.json') == test, 'Saved input manifest differs')
        verify_hashes({directory / name: digest for name, digest in result['snapshot_sha256'].items()})
        verify_hashes(result['model_asset_sha256'])
        assets = read(model_root / 'local_setup/manifest.json')
        for entry in [*assets['checkpoints'], *assets['text_assets']]:
            require(result['model_asset_sha256'][entry['path']] == entry['sha256'], 'Model/text asset differs from original manifest')
        image_hashes = read(directory / 'image_sha256.json')
        require(set(image_hashes) == {r['image'] for r in support['support_rows'] + test['rows']}, 'Incomplete image hashes')
        verify_hashes(image_hashes)
        require(not {image_hashes[r['image']] for r in support['support_rows']} & {image_hashes[r['image']] for r in test['rows']}, 'Exact-byte support/test overlap')
        integrity = read(directory / 'extension_integrity.json')
        require(integrity == result['extension_integrity'], 'Integrity record mismatch')
        require(integrity['original_75_queries_bitwise_equal'] is True and integrity['all_checkpoint_state_values_unchanged'] is True
                and integrity['state_sha256_before'] == integrity['state_sha256_after'], 'Original checkpoint/query context changed')
        require(integrity['query_context_count'] == len(expected_context), 'Wrong extended context size')
        cache_checks = integrity['pixel_cache_checks']
        require(any(check.get('passed') is True and check.get('native_preprocessing_bitwise_equal') is True
                    and check.get('max_abs_error') == 0.0 for check in cache_checks), 'MedKLIP cached native pixels differ')
        score_checks = [check for check in cache_checks if 'native_score_max_abs_error' in check]
        require({check['context'] for check in score_checks} == {'base75', integrity['active_mode']}, 'MedKLIP cache lacks original/extended-context score regression')
        require(all(check['passed'] is True and check['native_score_max_abs_error'] <= 1.2e-5
                    for check in score_checks), 'MedKLIP cached native scorer differs')
        last = integrity['context_history'][-1]
        require(last['classes_added'] == added_names and last['descriptions'] == [descriptions[name] for name in added_names], 'Different descriptions appended')
        fit = read(directory / 'query_adapter.json')
        original, refined = np.asarray(fit['original_queries']), np.asarray(fit['refined_queries'])
        with np.load(directory / 'query_context.npz', allow_pickle=False) as context:
            require(context['context_names'].tolist() == expected_context, 'Query context ordering changed')
            base, full = array(context, 'base_raw_queries'), array(context, 'full_raw_queries')
            require(base.shape == (75, 768) and full.shape == (len(expected_context), 768), 'Wrong raw query shape')
            np.testing.assert_array_equal(full[:75], base)
            projected = array(context, 'full_projected_queries')
            require(projected.shape == (len(expected_context), 256), 'Wrong projected context shape')
            np.testing.assert_array_equal(projected[expected_indices], original)
        parity = read(directory / 'native_parity.json')
        require(parity == result['native_parity'] and parity['passed'] is True and parity['max_abs_error'] <= 1.2e-5, 'Native path parity failed')
        regression = read(directory / 'supported_class_regression.json')
        require(regression == result['supported_class_regression'] and regression['passed'] is True and regression['images'] == 8
                and regression['query_context_count'] == 75 and regression['max_abs_error'] <= 4e-4, 'Original75 supported-class regression failed')
        reference = regression['reference_result']
        verify_hash(reference['result'], reference['result_sha256'])
        verify_hash(Path(reference['result']).parent / 'predictions.npz', regression['reference_predictions_sha256'])
    else:
        return verify_mavl_artifacts(directory, result, test, support, expected_context, expected_indices, added_names, added_keys)
    features = common.finite_array(directory / 'support_features.npy')
    labels = common.finite_array(directory / 'support_labels.npy')
    np.testing.assert_array_equal(labels, np.asarray([r['labels'] for r in support['support_rows']]))
    require(fit['class_names'] == test['class_names'] and fit['class_query_indices'] == expected_indices
            and fit['support_count'] == len(support['support_rows']), 'Support fit ordering changed')
    query = verify_support_queries(model, features, labels, original, refined, fit)
    cache_regression = None
    if result['dataset'] == 'montgomery':
        archived = HERE / 'medklip/archive/montgomery_uncached_complete'
        archived_result = read(archived / 'result.json')
        require(archived_result['status'] == 'complete' and archived_result['is_smoke'] is False
                and archived_result['rows'] == result['rows'] == 128, 'Missing full uncached Montgomery regression')
        for name in ('predictions.npz', 'support_features.npy', 'support_labels.npy'):
            verify_hash(archived / name, archived_result['artifact_sha256'][name])
            require(sha(archived / name) == sha(directory / name), 'Full cached/uncached MedKLIP arrays differ: ' + name)
        verify_hash(archived / 'result.json', sha(archived / 'result.json'))
        cache_regression = {'passed': True, 'rows': 128, 'all_prediction_and_support_arrays_bitwise_equal': True,
                            'uncached_result': str(archived / 'result.json'), 'uncached_result_sha256': sha(archived / 'result.json')}
    return {'passed': True, 'added_classes': added_names, 'context_count': len(expected_context),
            'query': query, 'extension_integrity': integrity, 'native_parity': parity,
            'original_context_regression': regression, 'full_cache_regression': cache_regression}


def verify_mavl_artifacts(directory, result, test, support, expected_context, expected_indices, added_names, added_keys):
    require(read(directory / 'manifest.json') == support, 'MAVL support manifest changed')
    inputs = read(directory / 'input_manifest.json')
    require(inputs['support_rows'] == support['support_rows'] and inputs['class_names'] == test['class_names'], 'MAVL saved inputs changed')
    require(common.row_identity(inputs['rows']) == common.row_identity(test['rows']), 'MAVL test rows changed')
    verify_hashes(inputs['source_sha256'])
    image_hashes = {r['image']: r['byte_sha256'] for r in inputs['support_rows'] + inputs['rows']}
    verify_hashes(image_hashes)
    require(not {r['byte_sha256'] for r in inputs['support_rows']} & {r['byte_sha256'] for r in inputs['rows']}, 'MAVL exact-byte overlap')
    for snapshot in read(directory / 'source_snapshots/index.json'):
        verify_hash(directory / snapshot['snapshot'], snapshot['sha256'])
        require(result['source_sha256'][snapshot['source']] == snapshot['sha256'], 'MAVL snapshot/source mismatch')
    config = read(directory / 'run_config.json')
    require(config['source_sha256'] == result['source_sha256'] and config['eta'] == .3
            and config['views'] == ['identity', 'center80'] and config['center_ratio'] == .8
            and config['probability_fusion'] == 'arithmetic mean', 'MAVL run configuration changed')
    require(config['query_class_names'] == expected_context and config['extended_query_context_classes'] == len(expected_context), 'MAVL context configuration changed')
    require('cuda.matmul.allow_tf32=False' in config['precision'] and 'cudnn.allow_tf32=True' in config['precision'], 'MAVL precision differs from original worker')
    pixel_cache = read(directory / 'pixel_cache_parity.json')
    require(pixel_cache['passed'] is True and pixel_cache['bitwise_equal'] is True
            and pixel_cache['max_abs_error'] == 0.0, 'MAVL cached native pixels differ')
    checkpoint = Path(config['native_checkpoint'])
    model_assets = read(ROOT / 'baselines/MAVL/local_setup/manifest.json')
    expected_weight = next(v['sha256'] for v in model_assets['checkpoints'] if Path(v['path']) == checkpoint)
    require(result['checkpoint_sha256'] == expected_weight, 'Wrong native MAVL checkpoint')
    verify_hash(checkpoint, expected_weight)
    canonical_sources = {str(Path(name).resolve()): digest for name, digest in result['source_sha256'].items()}
    for entry in model_assets['text_assets']:
        require(canonical_sources.get(str(Path(entry['path']).resolve())) == entry['sha256'], 'MAVL text assets differ from release')
        verify_hash(entry['path'], entry['sha256'])
    integrity = read(directory / 'extension_integrity.json')
    require(integrity == result['extension_integrity'], 'MAVL integrity record mismatch')
    for key in ('original_75_disease_queries_bitwise_equal', 'original_675_concept_queries_bitwise_equal', 'all_checkpoint_state_values_unchanged'):
        require(integrity[key] is True, f'MAVL context invariant failed: {key}')
    require(integrity['state_sha256_before'] == integrity['state_sha256_after'] and integrity['query_count'] == len(expected_context)
            and integrity['concepts_per_disease'] == 9 and integrity['new_class_order'] == added_keys
            and integrity['new_query_count'] == len(added_keys), 'MAVL state/context extension differs')
    descriptions = read(DESCRIPTIONS)['descriptions']
    require(result['original_adapter_metadata']['new_descriptions'] == {key: descriptions[name] for name, key in zip(added_names, added_keys)}, 'MAVL descriptions differ from frozen resource')
    fit = read(directory / 'query_adapter.json')
    require(fit == result['query_metadata'] and fit['class_names'] == test['class_names'] and fit['class_query_indices'] == expected_indices
            and fit['support_count'] == len(support['support_rows']), 'MAVL support fit order changed')
    with np.load(directory / 'queries.npz', allow_pickle=False) as queries:
        original, refined = array(queries, 'original'), array(queries, 'refined')
        np.testing.assert_array_equal(original, np.asarray(fit['original_queries']))
        np.testing.assert_array_equal(refined, np.asarray(fit['refined_queries']))
    with np.load(directory / 'query_context.npz', allow_pickle=False) as context:
        require(context['context_class_names'].tolist() == expected_context and context['target_class_indices'].tolist() == expected_indices, 'MAVL context/query indices changed')
        base, full = array(context, 'base_raw_queries'), array(context, 'full_raw_queries')
        concepts, full_concepts = array(context, 'base_concept_queries'), array(context, 'full_concept_queries')
        require(base.shape == (75, 768) and full.shape == (len(expected_context), 768), 'Wrong MAVL disease query shape')
        require(concepts.shape == (675, 768) and full_concepts.shape == (9*len(expected_context), 768), 'Wrong MAVL concept query shape')
        np.testing.assert_array_equal(full[:75], base)
        np.testing.assert_array_equal(full_concepts[:675], concepts)
        np.testing.assert_array_equal(full[75:], context['new_description_embeddings'])
        np.testing.assert_array_equal(full_concepts[675:], np.repeat(full[75:], 9, axis=0))
        projected = array(context, 'full_projected_queries')
        require(projected.shape == (len(expected_context), 9, 256), 'Wrong MAVL projected context shape')
        np.testing.assert_array_equal(projected[expected_indices], original)
        np.testing.assert_array_equal(projected[75:], np.repeat(projected[75:, :1], 9, axis=1))
    features = common.finite_array(directory / 'support_features.npy')
    labels = common.finite_array(directory / 'support_labels.npy')
    np.testing.assert_array_equal(labels, np.asarray([r['labels'] for r in support['support_rows']]))
    query = verify_support_queries('MAVL', features, labels, original, refined, fit)
    parity, zero = read(directory / 'native_parity.json'), read(directory / 'zero_eta_parity.json')
    require(parity == result['native_parity'], 'MAVL parity copy differs')
    for check in (parity, zero):
        require(check['passed'] is True and check['max_abs_error'] <= 1.2e-5, 'MAVL native/zero-eta parity failed')
    regression = read(directory / 'historical_regression.json')
    require(regression['passed'] is True and regression['rows'] == 8 and regression['context_diseases'] == 75
            and regression['context_concepts'] == 675, 'MAVL historical original context check missing')
    require(regression['pixel_cache']['passed'] is True and regression['pixel_cache']['bitwise_equal'] is True
            and regression['pixel_cache']['max_abs_error'] == 0.0, 'MAVL cached original-context pixels differ')
    verify_hash(regression['historical_result'], regression['historical_result_sha256'])
    verify_hash(Path(regression['historical_result']).parent / 'predictions.npz', regression['historical_predictions_sha256'])
    verify_hashes(regression['sources'])
    require(set(regression['branches']) == set(BRANCHES), 'MAVL historical branch regression incomplete')
    for check in regression['branches'].values():
        require(check['passed'] is True and check['max_abs_error'] <= 4e-4, 'MAVL historical prediction regression failed')
    for check in regression['query_checks'].values():
        require(check['max_abs_error'] <= 5e-5, 'MAVL historical query regression failed')
    return {'passed': True, 'added_classes': added_names, 'context_diseases': len(expected_context), 'concepts_per_disease': 9,
            'new_nine_slots_bitwise_equal': True, 'query': query, 'extension_integrity': integrity,
            'native_parity': parity, 'zero_eta_parity': zero, 'pixel_cache_parity': pixel_cache,
            'original_context_regression': regression}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=HERE / 'VERIFIED_COMPLETION.json')
    parser.add_argument('--audit-output', type=Path, default=HERE / 'verification_file_hashes.json')
    args = parser.parse_args()
    verify_hash(DESCRIPTIONS, DESCRIPTION_SHA)
    description = read(DESCRIPTIONS)
    require(description['fixed_before_inference'] is True, 'Descriptions were not frozen before inference')
    paths = [p.resolve() for p in (args.results or DEFAULT_RESULTS)]
    require(len(paths) == len(set(paths)) == 6, 'Require exactly six distinct result paths')
    require(all(path.is_file() for path in paths), 'Missing one or more full result.json files')
    results = [read(path) for path in paths]
    require({(r['model'], r['dataset']) for r in results} == EXPECTED, 'Missing, duplicate or unexpected pairs')
    for path, result in zip(paths, results):
        require(result.get('status') == 'complete' and result.get('is_smoke') is False and result.get('task') == 'classification', f'Smoke/incomplete/nonclassification result rejected: {path}')
    records = [verify_run(path, result) for path, result in zip(paths, results)]
    source_hashes = {str(Path(__file__).resolve()): sha(__file__), str(HELPER): sha(HELPER)}
    audit = {'status': 'verified', 'verifier_source_sha256': source_hashes,
             'file_count': len(common.VERIFIED_FILES), 'verified_file_sha256': dict(sorted(common.VERIFIED_FILES.items()))}
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    document = {'status': 'verified', 'prediction_metrics_verified': True, 'cpu_only': True,
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'verifier_source_sha256': source_hashes,
                'frozen_descriptions_sha256': DESCRIPTION_SHA, 'records': records,
                'file_hash_audit': str(args.audit_output.resolve()), 'file_hash_audit_sha256': sha(args.audit_output)}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(f'Verified all six pairs, 24 branches and {len(common.VERIFIED_FILES)} hashed files: {args.output}', flush=True)


if __name__ == '__main__':
    main()
