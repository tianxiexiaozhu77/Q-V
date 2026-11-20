#!/usr/bin/env python3
"""Independent audit of saved four-arm query-transfer evaluations (CPU only).

Reads only controller-selected full outputs. Does not import model adapters or
the inference/metric implementation, and never changes experimental artifacts.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
MODELS = {'BiomedCLIP', 'BioViL-T', 'CheXzero', 'MedKLIP', 'MAVL', 'KAD', 'MaCo', 'CARZero'}
EXPECTED_COUNTS = {'complete': 89, 'unsupported': 15}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


class Hashes:
    def __init__(self):
        self.cache = {}

    def digest(self, path):
        path = str(Path(path).resolve())
        if path not in self.cache:
            hasher = hashlib.sha256()
            with open(path, 'rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(block)
            self.cache[path] = hasher.hexdigest()
        return self.cache[path]

    def check(self, path, expected):
        require(self.digest(path) == expected, 'SHA256 changed: ' + str(path))

    def mapping(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), 'Empty source hash mapping')
        for path, expected in mapping.items():
            self.check(path, expected)


def close(actual, expected, context, atol=1e-12, rtol=0):
    if expected is None:
        require(actual is None, context + ': expected null')
    else:
        require(actual is not None and np.isfinite(actual), context + ': missing/nonfinite value')
        require(np.isclose(actual, expected, atol=atol, rtol=rtol),
                f'{context}: saved={actual}, recomputed={expected}')


def classification(labels, scores, names, recorded, context):
    require(scores.shape == labels.shape, context + ': prediction shape')
    require(np.isfinite(scores).all(), context + ': nonfinite scores')
    require(set(recorded['per_class']) == set(names), context + ': metric class names')
    values = {'auroc': [], 'auprc': [], 'average_precision': []}
    for index, name in enumerate(names):
        target, prediction = labels[:, index], scores[:, index]
        item = recorded['per_class'][name]
        require(item['positive_count'] == int(target.sum()), context + '/' + name + ': positive count')
        if len(np.unique(target)) != 2:
            for metric in values:
                close(item[metric], None, context + '/' + name + '/' + metric)
            continue
        precision, recall, _ = precision_recall_curve(target, prediction)
        trapz = getattr(np, 'trapezoid', None) or np.trapz
        actual = {'auroc': float(roc_auc_score(target, prediction)),
                  'auprc': float(trapz(precision[::-1], recall[::-1])),
                  'average_precision': float(average_precision_score(target, prediction))}
        for metric, value in actual.items():
            close(item[metric], value, context + '/' + name + '/' + metric)
            values[metric].append(value)
    require(recorded['class_count'] == len(names), context + ': class count')
    require(recorded['valid_class_count'] == len(values['auroc']), context + ': valid class count')
    for metric, items in values.items():
        close(recorded['macro_' + metric], float(np.mean(items)) if items else None, context + '/' + metric)


def spatial(items, rows, names, task, recorded, context):
    require(len(items) == len(rows), context + ': spatial row count')
    for image_index, (image, row) in enumerate(zip(items, rows)):
        require(len(image) == len(names), context + ': spatial class count')
        for class_index, name in enumerate(names):
            item = image[class_index]
            matches = [i for i, value in enumerate(row['class_names_present']) if value == name]
            require((item is not None) == bool(matches), context + ': positive spatial row mismatch')
            if item is None:
                continue
            require(np.isfinite(item['valid_fraction']) and 0 < item['valid_fraction'] <= 1,
                    context + ': invalid map coverage')
            if task == 'grounding':
                require(item['hit'] in (0, 1), context + ': invalid PG hit')
                xy = np.asarray(item['peak_xy'])
                require(xy.shape == (2,) and np.isfinite(xy).all() and (xy >= 0).all()
                        and np.array_equal(xy, xy.astype(np.int64)), context + ': invalid saved peak')
                x, y = xy
                hit = any(float(row['boxes'][i][0]) <= x <= float(row['boxes'][i][2])
                          and float(row['boxes'][i][1]) <= y <= float(row['boxes'][i][3]) for i in matches)
                require(item['hit'] == int(hit), f'{context}: saved peak/GT hit mismatch at row{image_index}')
            else:
                curve = np.asarray(item['dice_by_threshold'])
                require(curve.shape == (101,) and np.isfinite(curve).all()
                        and (curve >= 0).all() and (curve <= 1).all(), context + ': invalid Dice curve')
                require(item['positive_pixels'] > 0, context + ': no positive mask pixels')
                bounds = np.asarray(item['raw_minmax'])
                require(bounds.shape == (2,) and np.isfinite(bounds).all() and bounds[0] <= bounds[1],
                        context + ': invalid raw map bounds')
    metric = 'pointing_game' if task == 'grounding' else 'dice'
    scores, expected_classes = [], []
    for index, name in enumerate(names):
        valid = [image[index] for image in items if image[index] is not None]
        if not valid:
            continue
        expected_classes.append(name)
        saved = recorded['per_class'][name]
        require(saved['count'] == len(valid), context + '/' + name + ': positive image count')
        if task == 'grounding':
            value = float(np.mean([item['hit'] for item in valid]))
        else:
            curve = np.asarray([item['dice_by_threshold'] for item in valid]).mean(axis=0)
            best = int(curve.argmax())
            value = float(curve[best])
            close(saved['best_threshold'], float(np.arange(0, 1.01, .01)[best]), context + '/' + name + '/threshold')
            require(np.allclose(saved['mean_dice_curve'], curve, atol=1e-12, rtol=0), context + ': mean Dice curve')
        close(saved[metric], value, context + '/' + name + '/' + metric)
        scores.append(value)
    require(set(recorded['per_class']) == set(expected_classes), context + ': metric class set')
    require(recorded['valid_class_count'] == len(scores), context + ': spatial valid class count')
    close(recorded[metric], float(np.mean(scores)) if scores else None, context + '/' + metric)


def inspect_query_fit(meta, support, context):
    """Validate actual saved schemas: top-level stats, nested fit, or members."""
    names = support['class_names']
    labels = np.asarray([row['labels'] for row in support['support_rows']])
    expected_positive = labels.sum(axis=0)
    expected_negative = len(labels) - expected_positive
    require(meta['class_names'] == names and meta['support_count'] == len(labels), context + ': fit support identity')
    close(meta['eta'], .3, context + ': eta')
    for key in ('gradient_training', 'training'):
        if key in meta:
            require(meta[key] is False, context + ': gradient training unexpectedly enabled')
    containers = meta.get('members', [meta])
    require(isinstance(containers, list) and containers, context + ': empty query containers')
    if 'members' in meta:
        require(len(containers) == meta['checkpoint_count'], context + ': ensemble member count')
    findings = []
    for number, container in enumerate(containers):
        stats = container.get('fit', container)
        tag = context + f'/member{number}'
        direction = np.asarray(stats.get('directions', stats.get('direction')), dtype=np.float64)
        require(direction.ndim == 2 and direction.shape[0] == len(names) and np.isfinite(direction).all(),
                tag + ': direction shape/values')
        if 'direction' in stats and 'directions' in stats:
            require(np.array_equal(stats['direction'], stats['directions']), tag + ': direction alias mismatch')
        require(np.array_equal(stats['positive_counts'], expected_positive), tag + ': positive support counts')
        require(np.array_equal(stats['negative_counts'], expected_negative), tag + ': negative support counts')
        require((expected_positive > 0).all() and (expected_negative > 0).all(), tag + ': missing support polarity')
        positive = np.asarray(stats['positive_means'], dtype=np.float64)
        negative = np.asarray(stats['negative_means'], dtype=np.float64)
        require(positive.shape == negative.shape == direction.shape, tag + ': support mean shape')
        require(np.isfinite(positive).all() and np.isfinite(negative).all(), tag + ': nonfinite support means')
        difference = positive - negative
        length = np.linalg.norm(difference, axis=-1)
        require(np.allclose(stats['direction_norms'], length, atol=2e-6, rtol=2e-5), tag + ': direction difference norm')
        normalized = difference / np.maximum(length[:, None], 1e-12)
        require(np.allclose(direction, normalized, atol=3e-6, rtol=3e-5), tag + ': normalized positive-minus-negative direction')
        if 'zero_direction_classes' in stats:
            require(np.array_equal(stats['zero_direction_classes'], length < 1e-12), tag + ': zero direction flags')
        pairs = []
        for key in container:
            if key.startswith('original_') and key.endswith('queries'):
                refined_key = 'refined_' + key[len('original_'):]
                if refined_key in container:
                    pairs.append((key, refined_key))
        require(pairs, tag + ': no persisted original/refined query pair')
        pair_checks = []
        for original_key, refined_key in pairs:
            original = np.asarray(container[original_key], dtype=np.float64)
            refined = np.asarray(container[refined_key], dtype=np.float64)
            require(original.shape == refined.shape and original.ndim >= 2
                    and original.shape[0] == len(names) and original.shape[-1] == direction.shape[-1],
                    tag + ': query shape ' + original_key)
            require(np.isfinite(original).all() and np.isfinite(refined).all(), tag + ': query finite')
            norm = np.linalg.norm(original, axis=-1, keepdims=True)
            require((norm >= 1e-12).all(), tag + ': zero original query')
            expanded = direction.reshape((len(names),) + (1,) * (original.ndim - 2) + (direction.shape[-1],))
            expected = original / norm + .3 * expanded
            expected = expected / np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-12) * norm
            expected = np.where(np.linalg.norm(expanded, axis=-1, keepdims=True) < 1e-12, original, expected)
            require(np.allclose(refined, expected, atol=1e-5, rtol=3e-5), tag + ': query correction formula ' + original_key)
            require(np.allclose(np.linalg.norm(refined, axis=-1), norm[..., 0], atol=1e-5, rtol=3e-5),
                    tag + ': original query norms not preserved')
            pair_checks.append({'original_key': original_key, 'shape': list(original.shape),
                                'max_formula_absolute_error': float(np.abs(refined - expected).max())})
        findings.append({'member_index': container.get('member_index', number),
                         'direction_shape': list(direction.shape), 'query_checks': pair_checks})
    return findings


def audit(run, require_complete):
    hashes = Hashes()
    state_path = run / 'status.json'
    state = read(state_path)
    catalog_path = Path(state['catalog'])
    hashes.check(catalog_path, state['catalog_sha256'])
    catalog = read(catalog_path)
    hashes.check(catalog['test_catalog'], catalog['test_catalog_sha256'])
    hashes.mapping(catalog['source_catalogs'])
    old_catalog = read(catalog['test_catalog'])
    definitions = catalog['datasets']
    require(set(state['models']) == MODELS, 'Campaign must contain the expected eight models')
    require(len(definitions) == 13 and set(definitions) == set(old_catalog['datasets']), 'Changed evaluation scope')
    require(set(state['gpus']) <= {4, 5, 6, 7}, 'GPU scope differs from user authorization')
    manifests, supports = {}, {}
    for name, definition in definitions.items():
        old = old_catalog['datasets'][name]
        require(all(definition[key] == old[key] for key in ('manifest', 'sha256', 'count', 'class_names', 'task')),
                name + ': test definition differs from frozen baseline catalog')
        hashes.check(definition['manifest'], definition['sha256'])
        hashes.check(definition['support_manifest'], definition['support_manifest_sha256'])
        data, support = read(definition['manifest']), read(definition['support_manifest'])
        require(data['count'] == len(data['rows']) == definition['count'], name + ': manifest row count')
        require(data['class_names'] == support['class_names'] == definition['class_names'], name + ': manifest class order')
        require(support['test_manifest'] == definition['manifest'] and support['test_manifest_sha256'] == definition['sha256'],
                name + ': support/test manifest link')
        require(support['support_count'] == len(support['support_rows']) == definition['support_count'], name + ': support count')
        labels = np.asarray([row['labels'] for row in support['support_rows']])
        require(labels.shape == (support['support_count'], len(data['class_names']))
                and np.isin(labels, [0, 1]).all(), name + ': support labels')
        require(np.array_equal(labels.sum(0), support['support_positive_counts']), name + ': recorded support counts')
        hashes.mapping(support['support_sources'])
        test_paths = {str(Path(row['image']).resolve()) for row in data['rows']}
        support_paths = set()
        for row in support['support_rows']:
            hashes.check(row['image'], row['byte_sha256'])
            support_paths.add(str(Path(row['image']).resolve()))
        require(not support_paths & test_paths, name + ': support/test image path overlap')
        manifests[name], supports[name] = data, support
    checks, errors, workers = [], [], []
    for model, job in state['models'].items():
        folder = Path(job['full_output']) if job.get('full_output') else None
        if folder is None:
            checks.extend({'model': model, 'dataset': name, 'status': 'pending'} for name in definitions)
            continue
        try:
            attempts = [item for item in job['attempts'] if item['stage'] == 'full' and Path(item['output']) == folder]
            require(len(attempts) == 1 and '--smoke' not in attempts[0]['command'], model + ': output is not a unique full attempt')
            source_file = folder / 'source_sha256.json'
            sources = read(source_file)
            hashes.mapping(sources)
            worker_state = read(folder / 'status.json')
            if job['status'] == 'complete':
                require(worker_state['status'] == 'complete' and not worker_state['changed_sources'], model + ': worker not cleanly complete')
                require(attempts[0].get('returncode') == 0, model + ': worker exit status')
                require(set(worker_state['outcomes']) == set(definitions), model + ': worker scope')
            workers.append({'model': model, 'full_output': str(folder), 'status': job['status'],
                            'current_source_sha256_verified': sources})
        except Exception as error:
            errors.append({'model': model, 'stage': 'worker_provenance', 'error': repr(error)})
            sources = None
        for name, definition in definitions.items():
            path = folder / name / 'result.json'
            if not path.is_file():
                checks.append({'model': model, 'dataset': name, 'status': 'pending'})
                continue
            try:
                result = read(path)
                require(result['model'] == model and result['dataset'] == name and result['task'] == definition['task'], 'Result identity')
                require(result['rows'] == definition['count'], 'Full evaluation row count')
                reference = supports[name]['baseline_references'][model]
                hashes.check(reference['result'], reference['result_sha256'])
                old_result = read(reference['result'])
                if result['status'] == 'unsupported':
                    require(reference['status'] == old_result['status'] == 'unsupported', 'Newly unsupported task')
                    require(bool(result.get('reason')), 'Unsupported reason absent')
                    checks.append({'model': model, 'dataset': name, 'status': 'unsupported', 'reason': result['reason']})
                    continue
                require(result['status'] == reference['status'] == old_result['status'] == 'complete', 'Result failed/not complete')
                require(result['smoke'] is False, 'Smoke result cannot enter full audit')
                require(result['class_names'] == definition['class_names'], 'Result class order')
                require(result['support_count'] == definition['support_count'], 'Result support count')
                require(result['test_labels_used_for_query_fit'] is False and result['no_gradient_training'] is True, 'Fit protocol flags')
                close(result['eta'], .3, 'Result eta')
                require(result['baseline_reference'] == reference, 'Baseline reference mismatch')
                require(sources is not None and result['source_sha256'] == sources, 'Task/worker source provenance mismatch')
                require(read(path.with_name('manifest.json')) == supports[name], 'Saved support manifest differs')
                parity = result['native_parity']
                require(parity['passed'] is True and np.isfinite(parity['max_abs_error']), 'Native parity check failed')
                require(read(path.with_name('native_parity.json')) == parity, 'Native parity receipt mismatch')
                require(set(result['branches']) == set(BRANCHES), 'Missing/extra experimental branch')
                query_checks = inspect_query_fit(read(path.with_name('query_adapter.json')), supports[name], model + '/' + name)
                rows, names = manifests[name]['rows'], definition['class_names']
                context = model + '/' + name
                archive = {'result': reference['result'], 'result_sha256': reference['result_sha256']}
                if definition['task'] == 'classification':
                    prediction_file = path.with_name('predictions.npz')
                    archive_file = Path(reference['result']).with_name('predictions.npz')
                    with np.load(prediction_file, allow_pickle=False) as saved, np.load(archive_file, allow_pickle=False) as old:
                        require(set(saved.files) == set(BRANCHES) | {'labels', 'image_paths', 'class_names'}, 'Unexpected prediction keys')
                        expected_paths = [row['image'] for row in rows]
                        expected_labels = np.asarray([row['labels'] for row in rows])
                        require(saved['image_paths'].tolist() == old['image_paths'].tolist() == expected_paths, 'Image path identity/order')
                        require(saved['class_names'].tolist() == old['class_names'].tolist() == names, 'Saved class order')
                        require(np.array_equal(saved['labels'], expected_labels) and np.array_equal(old['labels'], expected_labels), 'Saved label identity')
                        require(expected_labels.shape == (len(rows), len(names)), 'Label shape')
                        require(np.isin(expected_labels, [0, 1]).all(), 'Nonbinary test labels')
                        for branch in BRANCHES:
                            classification(expected_labels, saved[branch], names, result['branches'][branch], context + '/' + branch)
                        old_scores = old['scores']
                        require(old_scores.shape == saved['original'].shape and np.isfinite(old_scores).all(), 'Archived score validity')
                        maximum = float(np.abs(saved['original'] - old_scores).max())
                        require(np.allclose(saved['original'], old_scores, atol=2e-4, rtol=2e-4), 'Archived original score parity')
                        archive.update(max_absolute_score_difference=maximum, atol=2e-4, rtol=2e-4,
                                       predictions_sha256=hashes.digest(archive_file))
                    primary = ('macro_auroc', 'macro_auprc')
                    prediction_sha = hashes.digest(prediction_file)
                else:
                    prediction_file = path.with_name('spatial_predictions.json')
                    saved = read(prediction_file)
                    require(saved['image_paths'] == [row['image'] for row in rows] and saved['class_names'] == names, 'Spatial identity/order/classes')
                    require(set(saved['branches']) == set(BRANCHES), 'Spatial branch set')
                    for branch in BRANCHES:
                        spatial(saved['branches'][branch], rows, names, definition['task'], result['branches'][branch], context + '/' + branch)
                    primary = ('pointing_game',) if definition['task'] == 'grounding' else ('dice',)
                    metric = primary[0]
                    close(result['branches']['original'][metric], old_result['metrics'][metric], 'Archived original spatial metric', atol=1e-6)
                    archive['max_absolute_primary_metric_difference'] = abs(result['branches']['original'][metric] - old_result['metrics'][metric])
                    prediction_sha = hashes.digest(prediction_file)
                require(set(result['delta_pp']) == set(primary), 'Gain metric keys')
                for metric in primary:
                    original, refined = result['branches']['original'][metric], result['branches']['tta_center_refined'][metric]
                    close(result['delta_pp'][metric], None if original is None else 100 * (refined - original), context + '/delta/' + metric)
                checks.append({'model': model, 'dataset': name, 'status': 'complete', 'rows': len(rows),
                               'four_branches_verified': list(BRANCHES), 'result_sha256': hashes.digest(path),
                               'predictions_sha256': prediction_sha, 'support_manifest_sha256': definition['support_manifest_sha256'],
                               'query_adapter_sha256': hashes.digest(path.with_name('query_adapter.json')),
                               'query_fit_checks': query_checks, 'archive_parity': archive})
            except Exception as error:
                checks.append({'model': model, 'dataset': name, 'status': 'failed_audit', 'error': repr(error)})
                errors.append({'model': model, 'dataset': name, 'error': repr(error), 'traceback': traceback.format_exc()})
    counts = dict(Counter(item['status'] for item in checks))
    fully_complete = counts == EXPECTED_COUNTS and state['status'] == 'complete'
    if require_complete and not fully_complete:
        errors.append({'stage': 'completion', 'error': 'Expected complete controller and89complete/15unsupported',
                       'controller_status': state['status'], 'counts': counts})
    return {'status': 'failed' if errors else ('passed' if fully_complete else 'partial_passed'),
            'checked_at': datetime.now(timezone.utc).isoformat(), 'run': str(run),
            'require_complete': require_complete, 'controller_status': state['status'],
            'counts': counts, 'expected_counts': EXPECTED_COUNTS, 'errors': errors,
            'catalog': str(catalog_path), 'catalog_sha256': hashes.digest(catalog_path),
            'auditor_sha256': hashes.digest(__file__), 'hashes_verified_count': len(hashes.cache),
            'workers': workers, 'checks': checks,
            'limits': ['Audit checks persisted predictions, identities, metrics, support/query arithmetic and current source hashes; it does not rerun networks.',
                       'PG saved peak/GT membership is recomputed; unsaved raw maps cannot be used to independently verify peak selection or view fusion.',
                       'Dice is independently aggregated from saved per-image101-threshold curves; raw masks/maps are not regenerated.',
                       'Support pixels are byte-hash verified. Test identity/labels follow frozen manifests; this audit does not rehash all test pixels or prove missing patient linkages.',
                       'Reduced-support protocols, native model preprocessing differences, and prior exploratory use of test sets remain unchanged.']}


def write(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    require(run.is_dir(), 'Run directory absent')
    try:
        report = audit(run, args.require_complete)
    except Exception as error:
        report = {'status': 'failed', 'run': str(run), 'checked_at': datetime.now(timezone.utc).isoformat(),
                  'require_complete': args.require_complete, 'counts': {},
                  'errors': [{'stage': 'global_provenance', 'error': repr(error), 'traceback': traceback.format_exc()}]}
    write(run / 'FINAL_AUDIT.json', report)
    write(ROOT / 'QUERY_REFINEMENT_AUDIT.json', report)
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'errors': len(report['errors'])}))
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
