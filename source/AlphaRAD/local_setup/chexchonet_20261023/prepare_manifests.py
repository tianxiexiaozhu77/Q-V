#!/usr/bin/env python3
"""Freeze custom CheXchoNet 7494 comparison splits without reading predictions.

Each task gets independently sampled 5-positive/5-negative patients (seed 42),
one eligible image per patient. All images from the union of support patients
are excluded from a common evaluation set used by all tasks and both models.
The input is a custom patient-sampled subset, not the AlphaRAD paper split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / 'data/alpharad/chexchonet-custom-test-7494'
TASKS = {
    'slvh': ('CSLV', 'Severe Left Ventricular Hypertrophy'),
    'dlv': ('CDLV', 'Dilated Left Ventricle'),
    'composite_slvh_dlv': ('CCOM', 'Severe Left Ventricular Hypertrophy or Dilated Left Ventricle'),
}
SEED = 42
SHOTS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'Refusing to overwrite a different frozen manifest: {path}')
    path.write_text(content)


def save_csv(path, rows):
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue()
    if path.exists() and path.read_bytes() != content.encode():
        raise RuntimeError(f'Refusing to overwrite a different frozen manifest: {path}')
    path.write_bytes(content.encode())


def counts(rows):
    return {
        'image_count': len(rows),
        'patient_count': len({r['patient_id'] for r in rows}),
        'labels': {task: {'positive': sum(r[task] for r in rows),
                          'negative': len(rows) - sum(r[task] for r in rows)}
                   for task in TASKS},
    }


def task_rows(rows, task):
    cls = TASKS[task][1]
    return [{**r, 'label': repr([r[task]]), 'class': repr([cls] if r[task] else [])} for r in rows]


def main():
    source = DATA / 'test_metadata.csv'
    with source.open(newline='') as handle:
        raw = list(csv.DictReader(handle))
    sampling = json.loads((DATA / 'sampling.json').read_text())
    assert sha(source) == sampling['output_sha256']['test_metadata.csv']
    assert sampling['official_split'] is False and sampling['alpharad_paper_split'] is False
    rows = []
    for r in raw:
        image = (DATA / r['image_path']).resolve()
        assert image.is_file() and DATA in image.parents
        assert all(r[t] in ('0', '1') for t in TASKS)
        rows.append({'image': str(image), 'image_path': str(image), 'patient_id': r['patient_id'],
                     **{t: int(r[t]) for t in TASKS}})
    assert len(rows) == 7494 and len({r['image'] for r in rows}) == 7494
    assert len({r['patient_id'] for r in rows}) == 2500

    supports, selected_patients, decisions = {}, set(), {}
    for task in TASKS:
        rng = random.Random(SEED)
        chosen, selected = [], set()
        decisions[task] = []
        for label in (1, 0):
            by_patient = defaultdict(list)
            for i, row in enumerate(rows):
                if row[task] == label and row['patient_id'] not in selected:
                    by_patient[row['patient_id']].append(i)
            patients = sorted(by_patient)
            rng.shuffle(patients)
            assert len(patients) >= SHOTS
            chosen_patients = patients[:SHOTS]
            for patient in chosen_patients:
                eligible = sorted(by_patient[patient], key=lambda i: rows[i]['image'])
                index = rng.choice(eligible)
                chosen.append(index)
                selected.add(patient)
            decisions[task].append({'label': label, 'eligible_patient_count': len(patients),
                                    'selected_patient_count': len(chosen_patients)})
        support = [rows[i] for i in chosen]
        assert len(support) == 10 and len(selected) == 10
        assert Counter(r[task] for r in support) == {1: 5, 0: 5}
        supports[task] = support
        selected_patients.update(selected)

    evaluation = [r for r in rows if r['patient_id'] not in selected_patients]
    excluded = [r for r in rows if r['patient_id'] in selected_patients]
    assert not selected_patients & {r['patient_id'] for r in evaluation}
    assert len(evaluation) + len(excluded) == len(rows)
    for task in TASKS:
        assert 0 < sum(r[task] for r in evaluation) < len(evaluation)

    save_csv(OUT / 'full_evaluation.csv', rows)
    save_csv(OUT / 'evaluation.csv', evaluation)
    save_csv(OUT / 'excluded_patients_images.csv', excluded)
    support_paths = {r['image'] for support in supports.values() for r in support}
    save_csv(OUT / 'support_union.csv', [r for r in rows if r['image'] in support_paths])

    datasets, full_datasets = {}, {}
    for task, (code, cls) in TASKS.items():
        dest = OUT / task
        dest.mkdir(exist_ok=True)
        for filename, records in [('full_evaluation.csv', rows), ('support.csv', supports[task]),
                                  ('evaluation.csv', evaluation)]:
            save_csv(dest / filename, task_rows(records, task))
        full = {'task': 'classification', 'dataset_key': 'chexchonet_' + task,
                'display_name': 'CheXchoNet ' + code + ' (custom subset)', 'class_names': [cls],
                'source_label_column': task, 'csv_path': str(dest / 'full_evaluation.csv'),
                'csv_sha256': sha(dest / 'full_evaluation.csv'), 'expected_count': len(rows),
                'actual_count': len(rows), 'unique_images': len(rows),
                'positive_count': [sum(r[task] for r in rows)],
                'scope': 'full custom 7494-image subset; original model only; not paper split'}
        full_datasets['chexchonet_' + task] = full
        datasets['chexchonet_' + task] = {
            **full, 'csv_path': str(dest / 'evaluation.csv'),
            'csv_sha256': sha(dest / 'evaluation.csv'), 'expected_count': len(evaluation),
            'actual_count': len(evaluation), 'unique_images': len(evaluation),
            'positive_count': [sum(r[task] for r in evaluation)],
            'support_csv_path': str(dest / 'support.csv'), 'support_csv_sha256': sha(dest / 'support.csv'),
            'support_count': 10, 'support_positive_count': [5], 'support_negative_count': [5],
            'original_csv_path': str(dest / 'full_evaluation.csv'),
            'original_csv_sha256': sha(dest / 'full_evaluation.csv'),
            'original_evaluation_count': len(rows), 'removed_from_original_evaluation': len(excluded),
            'scope': 'matched comparison excluding the union of all task support patients',
        }

    audit = {
        'status': 'metadata_and_files_present_verified',
        'dataset': 'CheXchoNet 1.0.0', 'split_name': 'chexchonet-custom-test-7494',
        'official_split': False, 'alpharad_paper_split': False,
        'seed': SEED, 'support_per_task': {'positive': SHOTS, 'negative': SHOTS},
        'sampling_algorithm': 'For each task independently use random.Random(42); positive then negative; sort eligible patient IDs, shuffle, take five distinct patients, then choose one random eligible image per selected patient in sorted-path order; negative selection excludes that task\'s selected positive patients. Exclude the union of support patients across all three tasks from every matched test.',
        'selection_used_predictions': False, 'label_source': 'original released binary columns, unchanged',
        'full': counts(rows), 'evaluation': counts(evaluation), 'excluded': counts(excluded),
        'support_per_task_counts': {t: counts(s) for t, s in supports.items()},
        'support_union_patient_count': len(selected_patients),
        'support_union_unique_image_count': len(support_paths),
        'patient_overlap_count': 0, 'sampling_decisions': decisions,
        'canonical_query_names': {t: c for t, (_, c) in TASKS.items()},
        'query_names_are_verified_author_prompt': False,
        'prompt_note': 'Fixed English semantic names for this experiment. Author CheXchoNet prompt definitions are absent from the public local repository, so equivalence to paper prompts is not asserted.',
        'limitations': ['The downloaded 7494-image subset is custom, even though the count equals the paper.',
                       'No independent train/support images are present locally; supports are reserved from this custom subset.',
                       'Image/patient isolation does not establish lack of overlap with pretrained-model source data.',
                       'Exact decoded-pixel duplicate audit is performed separately; near-duplicate images are not screened.',
                       'Full-set original results must not be subtracted from reduced-test adapted results.'],
        'official_description_url': 'https://physionet.org/content/chexchonet/1.0.0/',
        'source_files_sha256': {str(p): sha(p) for p in [source, DATA / 'sampling.json', DATA / 'README.txt', Path(__file__)]},
        'manifest_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*.csv'))},
    }
    write_json(OUT / 'split_audit.json', audit)
    for value in datasets.values():
        value['split_audit_path'] = str(OUT / 'split_audit.json')
        value['split_audit_sha256'] = sha(OUT / 'split_audit.json')
    protocol = {'seed': SEED, 'eta': 0.3, 'center_crop_ratio': 0.8, 'no_gradient_training': True,
                'baseline_and_refined_use_identical_evaluation_images': True,
                'all_models_and_tasks_use_identical_evaluation_images': True}
    write_json(OUT / 'catalog.json', {'status': 'complete', 'protocol': protocol, 'datasets': datasets})
    write_json(OUT / 'full_catalog.json', {'status': 'complete', 'datasets': full_datasets})
    print(json.dumps({'full': counts(rows), 'test': counts(evaluation), 'excluded': counts(excluded),
                      'support_union_patients': len(selected_patients), 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
