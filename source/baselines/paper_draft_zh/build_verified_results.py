#!/usr/bin/env python3
"""Collect published local experiment JSONs, without rerunning inference.

All metrics remain in their native [0, 1] scale. Delta columns are percentage
points. AUPRC is the trapezoidal area, not average precision (AP).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
METRICS = ('macro_auroc', 'macro_auprc', 'macro_average_precision', 'pointing_game', 'dice')
ORDER = ('AlphaRAD', 'RadZero', 'BiomedCLIP', 'BioViL-T', 'CheXzero', 'MedKLIP', 'MAVL', 'KAD', 'MaCo', 'CARZero')
# Authenticate the full historical collection before applying the manuscript's
# model and dataset scope to its numerical exports.
PAPER_MODELS = ORDER
PAPER_EXCLUDED_DATASETS = {'openi'}
COMPLETION_PATH = ROOT / 'experiments/table2_completion_20261103/VERIFIED_COMPLETION.json'
COMPLETION_PAIRS = {
    ('AlphaRAD', 'openi'), ('AlphaRAD', 'siim_radzero_holdout'),
    ('RadZero', 'chexpert'), ('RadZero', 'shenzhenxray'),
    ('RadZero', 'montgomery'), ('RadZero', 'covidkaggle'),
    ('RadZero', 'siim_pneumothorax'),
}
RESOURCE_COMPLETION_PATH = ROOT / 'experiments/query_resource_completion_20261103/VERIFIED_COMPLETION.json'
RESOURCE_COMPLETION_PAIRS = {
    (model, dataset) for model in ('MedKLIP', 'MAVL')
    for dataset in ('openi', 'shenzhenxray', 'montgomery')
}
RSNA_CLEANUP_PATH = ROOT / 'experiments/rsna_fixed_patient_cleanup_20261110/corrected_results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def primary(record):
    return ('macro_auroc', 'macro_auprc') if record['task'] == 'classification' else ('pointing_game',) if record['task'] == 'grounding' else ('dice',)


def classify_change(record):
    ds = [record['delta_pp'][m] for m in primary(record)]
    if all(x > 0 for x in ds):
        return 'all_primary_up'
    if all(x < 0 for x in ds):
        return 'all_primary_down'
    return 'mixed_or_tied'


def classification_means(rows, group_datasets=False):
    """Keep three correlated CheXchoNet endpoints from triple-weighting its data."""
    groups = {}
    for row in rows:
        key = row.get('dataset_group', row['dataset']) if group_datasets else row['dataset']
        groups.setdefault(key, []).append(row)
    dataset_means = {
        key: {b: {m: sum(r['branches'][b][m] for r in items) / len(items)
                  for m in ('macro_auroc', 'macro_auprc')} for b in BRANCHES}
        for key, items in groups.items()
    }
    means = {b: {m: sum(v[b][m] for v in dataset_means.values()) / len(groups)
                 for m in ('macro_auroc', 'macro_auprc')} for b in BRANCHES}
    return {
        'count': len(groups), 'task_count': len(rows), 'datasets': sorted(groups),
        'dataset_tasks': {key: [r['dataset'] for r in items] for key, items in groups.items()},
        'dataset_means': dataset_means, 'means': means,
        'delta_pp': {m: 100 * (means['tta_center_refined'][m] - means['original'][m])
                     for m in ('macro_auroc', 'macro_auprc')},
        'change_counts': dict(Counter(r['change'] for r in rows)),
        'change_counts_unit': 'individual task scopes, before dataset aggregation',
        'aggregation': ('equal dataset means; CheXchoNet three binary endpoints averaged first'
                        if group_datasets else 'equal individual classification task scopes'),
    }


def normalize(raw, model, dataset, source, catalog_entry):
    r = {
        'model': model, 'dataset': dataset, 'source_dataset': raw['dataset'],
        'task': raw['task'], 'status': raw['status'], 'rows': raw['rows'],
        'support_count': raw.get('support_count'),
        'class_names': raw.get('class_names') or catalog_entry['class_names'],
        'source_path': str(source), 'source_sha256': sha(source),
        'evaluation_manifest': catalog_entry['manifest'],
        'evaluation_manifest_sha256': catalog_entry['sha256'],
    }
    assert r['rows'] == catalog_entry['count'], (model, dataset, 'row count')
    if r['status'] != 'complete':
        r['reason'] = raw.get('reason', 'Unspecified')
        return r
    branches = raw.get('branches') or {k: v['metrics'] for k, v in raw['candidates'].items()}
    branches = {key: value.get('metrics', value) for key, value in branches.items()}
    if 'uniform_eta03' in branches:
        branches['query_refined'] = branches['uniform_eta03']
    assert all(b in branches for b in BRANCHES), (model, dataset, branches.keys())
    r['branches'] = {b: branches[b] for b in BRANCHES}
    r['delta_pp'] = {
        m: 100.0 * (branches['tta_center_refined'][m] - branches['original'][m])
        for m in METRICS if m in branches['original']
    }
    for b in BRANCHES:
        for m in primary(r):
            assert isinstance(branches[b][m], (int, float)) and math.isfinite(branches[b][m])
    r['change'] = classify_change(r)
    r['source_query_branch_name'] = 'uniform_eta03' if 'uniform_eta03' in branches else 'query_refined'
    r['query_norm_policy'] = ('normalize to unit norm' if model in ('AlphaRAD', 'RadZero') else 'preserve original query norm')
    r['eta'] = raw.get('eta', 0.3)
    return r


def summarize(records, models=ORDER, *, uniform_siim=False):
    # Historical collection checks retain their original aggregation. The
    # current manuscript uses the common 1,704-image protocol for every model.
    complete = [r for r in records if r['status'] == 'complete']
    result = {
        'counts': dict(Counter(r['status'] for r in records)),
        'complete_task_counts': dict(Counter(r['task'] for r in complete)),
        'classification_change_counts': dict(Counter(r['change'] for r in complete if r['task'] == 'classification')),
        'spatial_change_counts': dict(Counter(r['change'] for r in complete if r['task'] != 'classification')),
        'classification_means_including_two_siim_scopes': {},
        'classification_means_unique_datasets': {},
        'spatial_results': [],
        'negative_classification_results': [],
    }
    for model in models:
        selected = [r for r in complete if r['model'] == model and r['task'] == 'classification']
        if uniform_siim:
            siim = [r for r in selected if r['dataset'] == 'siim_pneumothorax']
            assert len(siim) == 1 and siim[0]['rows'] == 1704, model
        excluded_siim = ('siim_radzero_holdout' if uniform_siim or model != 'RadZero'
                         else 'siim_pneumothorax')
        for key, rows in (
            ('classification_means_including_two_siim_scopes', selected),
            ('classification_means_unique_datasets', [r for r in selected if r['dataset'] !=
                excluded_siim]),
        ):
            assert rows
            result[key][model] = classification_means(rows, group_datasets=key.endswith('unique_datasets'))
    for r in complete:
        compact = {k: r[k] for k in ('model', 'dataset', 'task', 'rows', 'delta_pp', 'change', 'source_path')}
        if r['task'] != 'classification':
            result['spatial_results'].append(compact)
        elif r['change'] != 'all_primary_up':
            result['negative_classification_results'].append(compact)
    return result


def completion_records(dataset_entries):
    """Append independently verified runs without changing the frozen catalog."""
    if not COMPLETION_PATH.exists():
        return []
    document = read(COMPLETION_PATH)
    assert document['status'] == 'verified'
    assert document['prediction_metrics_verified'] is True
    assert {(r['model'], r['dataset']) for r in document['records']} == COMPLETION_PAIRS
    assert len(document['records']) == len(COMPLETION_PAIRS)
    records = []
    for entry in document['records']:
        source = Path(entry['result'])
        assert sha(source) == entry['sha256'], ('completion result changed', source)
        raw = read(source)
        assert raw['status'] == 'complete' and raw['task'] == 'classification'
        assert (raw['model'], raw['dataset']) == (entry['model'], entry['dataset'])
        catalog_entry = dataset_entries[entry['dataset']]
        assert raw['class_names'] == catalog_entry['class_names']
        record = normalize(raw, entry['model'], entry['dataset'], source, catalog_entry)
        record['completion_batch'] = 'table2_completion_20261103'
        record['prediction_audit'] = entry['prediction_audit']
        records.append(record)
    return records


def apply_resource_completion(records, dataset_entries):
    """Replace six unavailable scopes with authenticated extended-query runs.

    Original source files and the frozen catalog are never changed. Each new
    record retains the unavailable record it supersedes for provenance.
    """
    if not RESOURCE_COMPLETION_PATH.exists():
        return records, []
    document = read(RESOURCE_COMPLETION_PATH)
    assert document['status'] == 'verified'
    assert document['prediction_metrics_verified'] is True
    assert len(document['records']) == len(RESOURCE_COMPLETION_PAIRS)
    assert {(r['model'], r['dataset']) for r in document['records']} == RESOURCE_COMPLETION_PAIRS
    by_pair = {(r['model'], r['dataset']): r for r in records}
    replaced = []
    for entry in document['records']:
        pair = (entry['model'], entry['dataset'])
        previous = by_pair[pair]
        assert previous['status'] == 'unsupported' and previous['task'] == 'classification'
        source = Path(entry['result'])
        assert sha(source) == entry['sha256'], ('resource completion result changed', source)
        raw = read(source)
        assert raw['status'] == 'complete' and raw['task'] == 'classification'
        assert (raw['model'], raw['dataset']) == pair
        assert not raw.get('is_smoke', False)
        catalog_entry = dataset_entries[entry['dataset']]
        assert raw['class_names'] == catalog_entry['class_names']
        record = normalize(raw, *pair, source, catalog_entry)
        record['completion_batch'] = 'query_resource_completion_20261103'
        record['prediction_audit'] = entry['prediction_audit']
        record['supersedes_unavailable_record'] = previous
        by_pair[pair] = record
        replaced.append(pair)
    assert len(by_pair) == len(records)
    return list(by_pair.values()), replaced


def apply_rsna_patient_cleanup(records):
    """Replace only fixed RSNA scores after authenticating historical inputs."""
    correction = read(RSNA_CLEANUP_PATH)
    assert correction['status'] == 'verified'
    separation = correction['summary']['separation']
    assert separation['original_test_images'] == 5327
    assert separation['excluded_test_images'] == 30
    assert separation['corrected_test_images'] == 5297
    assert separation['corrected_support_test_patient_overlap'] == 0
    replacements = {(r['model'], r['dataset']): r for r in correction['records']}
    assert len(replacements) == len(PAPER_MODELS)
    assert set(replacements) == {(model, 'rsna_pneumonia') for model in PAPER_MODELS}
    output = []
    replaced = set()
    for old in records:
        key = (old['model'], old['dataset'])
        if key not in replacements:
            output.append(old)
            continue
        new = replacements[key]
        historical = new['postprocessing']['historical_result']
        assert historical == {'path': old['source_path'], 'sha256': old['source_sha256']}
        assert old['rows'] == 5327 and new['rows'] == 5297
        for field in ('model', 'dataset', 'task', 'status', 'support_count', 'class_names'):
            assert new[field] == old[field], (key, field)
        assert sha(new['source_path']) == new['source_sha256']
        source = read(new['source_path'])
        assert source['branches'] == new['branches'] and source['rows'] == new['rows']
        assert sha(new['evaluation_manifest']) == new['evaluation_manifest_sha256']
        assert read(new['evaluation_manifest'])['count'] == new['rows']
        for metric in primary(new):
            delta = 100 * (new['branches']['tta_center_refined'][metric]
                           - new['branches']['original'][metric])
            assert abs(delta - new['delta_pp'][metric]) < 1e-12
        assert new['change'] == classify_change(new)
        output.append(new)
        replaced.add(key)
    assert replaced == set(replacements)
    return output


def main():
    catalog_path = ROOT / 'baselines/evaluation/matched_phase6_20261022/catalog.json'
    catalog = read(catalog_path)
    dataset_entries = catalog['datasets']
    aggregate_path = ROOT / 'baselines/QUERY_REFINEMENT_RESULTS.json'
    aggregate = read(aggregate_path)
    assert aggregate['status'] == 'complete'
    records = []
    for raw in aggregate['records']:
        source = Path(raw['result_file'])
        source_raw = read(source)
        assert source_raw['status'] == raw['status']
        if raw['status'] == 'complete':
            assert source_raw['branches'] == raw['branches'], (raw['model'], raw['dataset'], 'aggregate/source mismatch')
        records.append(normalize(raw, raw['model'], raw['dataset'], source, dataset_entries[raw['dataset']]))
    references = []
    seen = set()
    for dataset, entry in dataset_entries.items():
        assert sha(entry['manifest']) == entry['sha256'], ('manifest changed', dataset)
        for ref in entry['comparison_references']:
            source = Path(ref['result'])
            assert str(source) not in seen, ('duplicate source', str(source))
            seen.add(str(source))
            assert sha(source) == ref['sha256'], ('reference changed', str(source))
            raw = read(source)
            assert raw['status'] == 'complete'
            records.append(normalize(raw, ref['model'], dataset, source, entry))
            references.append(str(source))
    chexcho_path = OUT / 'CHEXCHONET_VERIFIED.json'
    chexcho = read(chexcho_path)
    assert chexcho['status'] == 'verified' and len(chexcho['records']) == 30
    for path, expected in chexcho['source_files_sha256'].items():
        assert sha(path) == expected, ('CheXchoNet source changed', path)
    records.extend(chexcho['records'])
    records.sort(key=lambda r: (ORDER.index(r['model']), r['dataset']))
    assert len(records) == 150
    assert len({(r['model'], r['dataset']) for r in records}) == len(records)
    summary = summarize(records)
    assert summary['counts'] == {'complete': 135, 'unsupported': 15}
    assert summary['complete_task_counts'] == {'classification': 117, 'grounding': 12, 'segmentation': 6}
    assert sum(m['count'] for m in summary['classification_means_unique_datasets'].values()) == 89
    for model in ORDER:
        expected = chexcho['summary_after_merge_with_verified_results']['classification_means_unique_datasets'][model]
        actual = summary['classification_means_unique_datasets'][model]
        assert actual['count'] == expected['count']
        for branch in BRANCHES:
            for metric in ('macro_auroc', 'macro_auprc'):
                assert abs(actual['means'][branch][metric] - expected['means'][branch][metric]) < 1e-12
    # Validate the historical 135-run collection above before adding new runs.
    additions = completion_records(dataset_entries)
    assert not ({(r['model'], r['dataset']) for r in records} &
                {(r['model'], r['dataset']) for r in additions})
    records.extend(additions)
    records, resource_replacements = apply_resource_completion(records, dataset_entries)
    records.sort(key=lambda r: (ORDER.index(r['model']), r['dataset']))
    summary = summarize(records)
    assert summary['counts']['complete'] == 135 + len(additions) + len(resource_replacements)
    assert summary['counts']['unsupported'] == 15 - len(resource_replacements)
    assert summary['complete_task_counts']['classification'] == 117 + len(additions) + len(resource_replacements)
    unique_count = sum(m['count'] for m in summary['classification_means_unique_datasets'].values())
    unique_task_count = sum(m['task_count'] for m in summary['classification_means_unique_datasets'].values())
    assert unique_count == (94 if additions else 89) + len(resource_replacements)
    records = [r for r in records if r['model'] in PAPER_MODELS
               and r['dataset'] not in PAPER_EXCLUDED_DATASETS]
    resource_replacements = [pair for pair in resource_replacements
                             if pair[0] in PAPER_MODELS and pair[1] not in PAPER_EXCLUDED_DATASETS]
    records = apply_rsna_patient_cleanup(records)
    summary = summarize(records, models=PAPER_MODELS, uniform_siim=True)
    unique_count = sum(m['count'] for m in summary['classification_means_unique_datasets'].values())
    unique_task_count = sum(m['task_count'] for m in summary['classification_means_unique_datasets'].values())
    complete_count = summary['counts']['complete']
    class_count = summary['complete_task_counts']['classification']
    model_counts = Counter(r['model'] for r in records if r['status'] == 'complete')
    document = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'verified',
        'metric_unit': 'fraction (0-1); delta_pp is percentage points',
        'auprc_definition': 'trapezoidal PR-curve area; average precision is stored separately',
        'primary_method': 'tta_center_refined', 'eta': 0.3,
        'branches': list(BRANCHES),
        'included_models': list(PAPER_MODELS),
        'included_datasets': sorted({r['dataset'] for r in records}),
        'source_aggregate': str(aggregate_path), 'source_aggregate_sha256': sha(aggregate_path),
        'source_catalog': str(catalog_path), 'source_catalog_sha256': sha(catalog_path),
        'chexchonet_verified_source': str(chexcho_path), 'chexchonet_verified_source_sha256': sha(chexcho_path),
        'chexchonet_protocol': chexcho['protocol'],
        'chexchonet_limitations': chexcho['limitations'],
        'completion_verified_source': str(COMPLETION_PATH) if additions else None,
        'completion_verified_source_sha256': sha(COMPLETION_PATH) if additions else None,
        'resource_completion_verified_source': str(RESOURCE_COMPLETION_PATH) if resource_replacements else None,
        'resource_completion_verified_source_sha256': sha(RESOURCE_COMPLETION_PATH) if resource_replacements else None,
        'rsna_patient_cleanup_source': str(RSNA_CLEANUP_PATH),
        'rsna_patient_cleanup_source_sha256': sha(RSNA_CLEANUP_PATH),
        'notes': [
            f'{complete_count} completed paired tasks: {sum(model_counts[m] for m in PAPER_MODELS if m not in ("AlphaRAD", "RadZero"))} transfers across {len(PAPER_MODELS) - 2} additional baselines, {model_counts["AlphaRAD"]} AlphaRAD tasks, {model_counts["RadZero"]} RadZero tasks.',
            f'The {summary["counts"]["unsupported"]} unsupported rows are not failed runs and are never assigned numerical scores.',
            f'{len(resource_replacements)} formerly unavailable classification scopes use fixed study-defined query descriptions; all four branches share the extended context, and historical unavailable records are retained as provenance.',
            'SIIM has two evaluation scopes (1704 and 1694 images); primary unique-dataset means uniformly retain the 1704-image protocol for every model. Both protocols remain in the per-protocol results.',
            f'CheXchoNet CSLV/CDLV/CCOM are three endpoints of one dataset; average their metrics before equal-dataset model means. {class_count} classification scopes, {unique_task_count} excluding duplicate SIIM scopes, {unique_count} model-by-dataset pairs.',
            'All source result hashes referenced by the frozen matched catalog were checked; every historically aggregated branch was checked against its individual result JSON before selecting the current manuscript models and datasets.',
            'Fixed-support RSNA scores use 5,297 test images after excluding 30 images from seven support patients using the official RSNA-to-NIH mapping. All ten models retain their original supports and predictions; only test scoring changes. Historical results and the separate 5,327-image repeated-support/path protocols are preserved.',
            'Earlier phase6 trials were exploratory; these numbers do not imply untouched test evaluation or statistical significance.',
        ],
        'summary': summary, 'records': records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'verified_results.json').write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
    fields = ['model', 'dataset', 'dataset_group', 'source_dataset', 'task', 'status', 'rows', 'support_count', 'class_count']
    fields += [f'{b}__{m}' for b in BRANCHES for m in METRICS]
    fields += [f'delta_pp__{m}' for m in METRICS]
    fields += ['change', 'reason', 'source_path', 'source_sha256', 'evaluation_manifest', 'evaluation_manifest_sha256']
    with (OUT / 'verified_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, '') for k in fields if k in r}
            row['class_count'] = len(r['class_names'])
            if r['status'] == 'complete':
                for b in BRANCHES:
                    for m in METRICS:
                        row[f'{b}__{m}'] = r['branches'][b].get(m, '')
                for m in METRICS:
                    row[f'delta_pp__{m}'] = r['delta_pp'].get(m, '')
            writer.writerow(row)
    print(json.dumps({k: summary[k] for k in ('counts', 'complete_task_counts', 'classification_change_counts', 'spatial_change_counts')}, indent=2))


if __name__ == '__main__':
    main()
