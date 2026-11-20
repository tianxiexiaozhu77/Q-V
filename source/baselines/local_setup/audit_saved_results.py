#!/usr/bin/env python3
"""Audit persisted predictions against the frozen test identities and metrics."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, context):
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f'{context}: saved={actual}, recalculated={expected}')


def audit(campaign, catalog, require_complete):
    state = json.loads(campaign.read_text())
    if require_complete:
        assert state['status'] == 'complete', 'Campaign is still running'
    definitions = json.loads(catalog.read_text())['datasets']
    manifests = {}
    for key, definition in definitions.items():
        assert digest(definition['manifest']) == definition['sha256'], key
        manifests[key] = json.loads(Path(definition['manifest']).read_text())
    checks = []
    siim_subset_checks = []
    for model, job in state['models'].items():
        folder = job.get('full_output')
        if not folder:
            checks.append({'model': model, 'status': 'pending'})
            continue
        folder = Path(folder)
        if job['status'] == 'complete':
            worker = json.loads((folder / 'status.json').read_text())
            assert worker['status'] == 'complete' and not worker['failed'], model
            assert worker['source_hashes_unchanged'], model
            for path, expected in json.loads((folder / 'source_sha256.json').read_text()).items():
                assert digest(path) == expected, path
        for key, definition in definitions.items():
            path = folder / key / 'result.json'
            if not path.exists():
                assert not require_complete, str(path)
                continue
            result = json.loads(path.read_text())
            assert result['model'] == model and result['dataset'] == key
            assert result['class_names'] == definition['class_names']
            assert result['expected_rows'] == definition['count']
            if result['status'] == 'unsupported':
                assert result.get('reason'), str(path)
                checks.append({'model': model, 'dataset': key, 'status': 'unsupported', 'reason': result['reason']})
                continue
            assert result['status'] == 'complete', str(path)
            assert result['rows'] == definition['count'], str(path)
            assert result['manifest_sha256'] == definition['sha256'], str(path)
            assert result['support_examples_used'] == 0
            assert result['network_weight_updates'] is False
            manifest = manifests[key]
            names = definition['class_names']
            if definition['task'] == 'classification':
                with np.load(path.with_name('predictions.npz'), allow_pickle=False) as saved:
                    labels, scores = saved['labels'], saved['scores']
                    assert scores.shape == labels.shape == (definition['count'], len(names))
                    assert np.isfinite(scores).all()
                    assert saved['class_names'].tolist() == names
                    assert saved['image_paths'].tolist() == [row['image'] for row in manifest['rows']]
                    assert np.array_equal(labels, np.asarray([row['labels'] for row in manifest['rows']]))
                    aucs, prs, aps = [], [], []
                    for c, name in enumerate(names):
                        y, p = labels[:, c], scores[:, c]
                        assert len(np.unique(y)) == 2, (model, key, name)
                        precision, recall, _ = precision_recall_curve(y, p)
                        aucs.append(float(roc_auc_score(y, p)))
                        prs.append(float(np.trapz(precision[::-1], recall[::-1])))
                        aps.append(float(average_precision_score(y, p)))
                        item = result['metrics']['per_class'][name]
                        assert item['positive_count'] == int(y.sum())
                        for metric, values in [('auroc', aucs), ('auprc', prs), ('average_precision', aps)]:
                            close(item[metric], values[-1], f'{model}/{key}/{name}/{metric}')
                    for metric, values in [('auroc', aucs), ('auprc', prs), ('average_precision', aps)]:
                        close(result['metrics']['macro_' + metric], np.mean(values), f'{model}/{key}/{metric}')
                    assert result['metrics']['valid_class_count'] == len(names)
            else:
                items = json.loads(path.with_name('spatial_predictions.json').read_text())
                assert len(items) == definition['count']
                for image, row in zip(items, manifest['rows']):
                    assert len(image) == len(names)
                    for c, name in enumerate(names):
                        assert (image[c] is not None) == (name in row['class_names_present'])
                class_scores = []
                metric = 'pointing_game' if definition['task'] == 'grounding' else 'dice'
                for c, name in enumerate(names):
                    valid = [row[c] for row in items if row[c] is not None]
                    saved_class = result['metrics']['per_class'][name]
                    assert saved_class['count'] == len(valid) and valid
                    if metric == 'pointing_game':
                        assert all(row['hit'] in (0, 1) for row in valid)
                        value = np.mean([row['hit'] for row in valid])
                    else:
                        curves = np.asarray([row['dice_by_threshold'] for row in valid])
                        assert curves.shape == (len(valid), 101)
                        assert np.isfinite(curves).all() and (curves >= 0).all() and (curves <= 1).all()
                        mean = curves.mean(axis=0)
                        value = mean.max()
                        close(saved_class['best_threshold'], np.arange(0, 1.01, .01)[mean.argmax()], f'{model}/{key}/threshold')
                    close(saved_class[metric], value, f'{model}/{key}/{name}/{metric}')
                    class_scores.append(value)
                close(result['metrics'][metric], np.mean(class_scores), f'{model}/{key}/{metric}')
            checks.append({'model': model, 'dataset': key, 'status': 'complete',
                           'rows': result['rows'], 'result_sha256': digest(path)})
        # SIIM's reduced split must be a literal image/label subset of the
        # complete split. Batch regrouping can introduce small FP32 differences.
        full_siim = folder / 'siim_pneumothorax/predictions.npz'
        subset_siim = folder / 'siim_radzero_holdout/predictions.npz'
        if job['status'] == 'complete' and full_siim.exists() and subset_siim.exists():
            with np.load(full_siim, allow_pickle=False) as full, np.load(subset_siim, allow_pickle=False) as subset:
                positions = {path: i for i, path in enumerate(full['image_paths'].tolist())}
                order = [positions[path] for path in subset['image_paths'].tolist()]
                assert np.array_equal(full['labels'][order], subset['labels'])
                maximum = float(np.abs(full['scores'][order] - subset['scores']).max())
                assert np.allclose(full['scores'][order], subset['scores'], atol=2e-4, rtol=2e-4), (model, maximum)
                siim_subset_checks.append({'model': model, 'subset_rows': len(order), 'max_absolute_score_difference': maximum})
    counts = dict(Counter(row['status'] for row in checks))
    if require_complete:
        assert counts == {'complete': 89, 'unsupported': 15}, counts
    return {'status': 'passed' if require_complete else 'partial_passed',
            'checked_at': datetime.now(timezone.utc).isoformat(), 'campaign': str(campaign),
            'catalog_sha256': digest(catalog), 'counts': counts,
            'checks': checks, 'siim_subset_checks': siim_subset_checks,
            'limits': 'Audits persisted scores, labels, identity/order, classes, metrics, source hashes and completion. Does not make distinct native preprocessing protocols identical.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT / 'CAMPAIGN_STATUS.json')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'evaluation/matched_phase6_20261022/catalog.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'FINAL_AUDIT.json')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = audit(args.campaign, args.catalog, args.require_complete)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': report['status'], 'counts': report['counts']}))


if __name__ == '__main__':
    main()
