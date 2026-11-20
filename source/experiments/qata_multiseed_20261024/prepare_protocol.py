#!/usr/bin/env python3
"""Freeze five QaTa support draws while preserving the existing validation/test.

This CPU-only preparation reads images and metadata, never model predictions.
Positive supports are sampled by known patient; the documented normal control
source has no patient mapping and is sampled by exact decoded-pixel group.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/qata_validation_threshold_20261024/protocol_v1/protocol.json'
DATA = ROOT / 'data/alpharad/QaTaCOV19/QaTa-COV19'
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = ('original', 'positive_query', 'query_refined',
            'tta_center_original', 'tta_center_refined')
IDENTITY_FIELDS = ('image', 'image_path', 'image_id', 'patient_id', 'group_id',
                   'image_size_wh', 'image_sha256', 'pixel_sha256')
PRIMARY_CONTRASTS = [('query_refined', 'original'), ('query_refined', 'positive_query')]
SECONDARY_CONTRASTS = [('positive_query', 'original'), ('tta_center_original', 'original'),
                       ('tta_center_refined', 'original'), ('tta_center_refined', 'query_refined'),
                       ('tta_center_refined', 'tta_center_original')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def checked(spec):
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Frozen file changed: ' + spec['path'])
    return json.loads(Path(spec['path']).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def patient_id(path):
    match = re.match(r'^(sub-S\d+)_', Path(path).name)
    return match.group(1) if match else None


def image_record(path):
    path = Path(path).resolve()
    with Image.open(path) as handle:
        rgb = handle.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + b'\0RGB\0' + rgb.tobytes()).hexdigest()
        size = list(rgb.size)
    return {'image': str(path), 'image_path': str(path), 'image_id': path.name,
            'patient_id': patient_id(path), 'group_id': patient_id(path),
            'image_size_wh': size, 'image_sha256': sha(path), 'pixel_sha256': pixel}


def verify_frozen_images(rows, workers):
    """Verify identities against the old manifest without reading any mask."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        current = list(pool.map(image_record, [r['image'] for r in rows]))
    for before, now in zip(rows, current, strict=True):
        if any(before[k] != now[k] for k in IDENTITY_FIELDS):
            raise ValueError('Frozen evaluation image changed: ' + before['image'])


def eligible_candidates(positive, negative, evaluation_rows):
    """Remove held-out patients/pixels and deterministically deduplicate images."""
    blocked_patients = {r['patient_id'] for r in evaluation_rows if r['patient_id']}
    blocked_pixels = {r['pixel_sha256'] for r in evaluation_rows}
    if any(not r['patient_id'] for r in positive):
        raise ValueError('All positive candidates must have a known patient ID')
    if any(r['patient_id'] is not None for r in negative):
        raise ValueError('Control_Group_I has no verified patient mapping')
    conflicts = {r['pixel_sha256'] for r in positive} & {r['pixel_sha256'] for r in negative}
    pixel_blocked_patients = {r['patient_id'] for r in positive
                              if r['pixel_sha256'] in blocked_pixels | conflicts}
    pos = [r for r in positive if r['patient_id'] not in blocked_patients | pixel_blocked_patients]
    neg = [r for r in negative if r['pixel_sha256'] not in blocked_pixels | conflicts]

    def unique_pixels(rows):
        representatives = {}
        for row in sorted(rows, key=lambda r: r['image']):
            representatives.setdefault(row['pixel_sha256'], row)
        return sorted(representatives.values(), key=lambda r: r['image'])

    pos_unique, neg_unique = unique_pixels(pos), unique_pixels(neg)
    audit = {
        'raw_positive_images': len(positive),
        'raw_positive_patients': len({r['patient_id'] for r in positive}),
        'raw_negative_images': len(negative),
        'blocked_evaluation_patients': sorted(blocked_patients),
        'positive_patients_excluded_by_patient_overlap': sorted({r['patient_id'] for r in positive} & blocked_patients),
        'positive_patients_excluded_by_pixel_overlap_or_label_conflict': sorted(pixel_blocked_patients),
        'cross_label_pixel_conflicts_excluded': sorted(conflicts),
        'positive_rows_removed_by_patient_exclusions': len(positive) - len(pos),
        'negative_rows_removed_by_pixel_exclusions': len(negative) - len(neg),
        'positive_exact_duplicate_rows_removed': len(pos) - len(pos_unique),
        'negative_exact_duplicate_rows_removed': len(neg) - len(neg_unique),
        'eligible_positive_images': len(pos_unique),
        'eligible_positive_patients': len({r['patient_id'] for r in pos_unique}),
        'eligible_negative_pixel_groups': len(neg_unique),
    }
    return pos_unique, neg_unique, audit


def sample_support(positive, negative, seed, per_class=5):
    """Independent seeded draws; select patients, then one image per patient."""
    if per_class < 1:
        raise ValueError('Support size must be positive')
    patients = defaultdict(list)
    for row in sorted(positive, key=lambda r: r['image']):
        patients[row['patient_id']].append(row)
    if len(patients) < per_class or len(negative) < per_class:
        raise ValueError('Insufficient eligible positive patients or negative pixel groups')
    rng = np.random.default_rng(seed)
    chosen_patients = sorted(rng.choice(sorted(patients), per_class, replace=False).tolist())
    selected = [patients[p][int(rng.integers(len(patients[p])))] for p in chosen_patients]
    negatives = sorted(negative, key=lambda r: r['image'])
    indices = rng.choice(len(negatives), per_class, replace=False)
    selected += sorted([negatives[int(i)] for i in indices], key=lambda r: r['image'])
    if len({r['pixel_sha256'] for r in selected}) != 2 * per_class:
        raise ValueError('Selected supports contain an exact decoded-pixel duplicate')
    return [dict(r, support_seed=int(seed)) for r in selected]


def validate_support(rows, evaluation_rows, per_class=5):
    labels = [r['label_vector'][0] for r in rows]
    if labels.count(1) != per_class or labels.count(0) != per_class:
        raise ValueError('Expected exactly five positive and five negative support images')
    positive = [r for r in rows if r['label_vector'] == [1]]
    if len({r['patient_id'] for r in positive}) != per_class or any(not r['patient_id'] for r in positive):
        raise ValueError('Positive supports must be different known patients')
    held_patients = {r['patient_id'] for r in evaluation_rows if r['patient_id']}
    held_pixels = {r['pixel_sha256'] for r in evaluation_rows}
    if any(r['patient_id'] in held_patients or r['pixel_sha256'] in held_pixels for r in rows):
        raise ValueError('Support intersects held-out patients or decoded pixels')
    if len({r['image'] for r in rows}) != len(rows) or len({r['pixel_sha256'] for r in rows}) != len(rows):
        raise ValueError('Support images/pixels are not distinct')


def prepare(output, source=SOURCE, workers=8):
    source = Path(source).resolve()
    original = json.loads(source.read_text())
    if original['status'] != 'frozen' or original['dataset'] != 'qatacovid_seg':
        raise ValueError('Expected the frozen QaTa validation-threshold protocol')
    fixed = {}
    for split, count, patients in [('validation', 473, 200), ('test', 2113, 474)]:
        entry = original[split]
        fixed[split] = checked({'path': entry['rows_path'], 'sha256': entry['rows_sha256']})
        if sha(entry['csv_path']) != entry['csv_sha256']:
            raise ValueError('Frozen CSV changed: ' + entry['csv_path'])
        rows = fixed[split]
        if len(rows) != count or len({r['patient_id'] for r in rows}) != patients or any(not r['patient_id'] for r in rows):
            raise ValueError('Unexpected frozen split size or missing patient IDs')
        verify_frozen_images(rows, workers)
    all_fixed = fixed['validation'] + fixed['test']
    if {r['patient_id'] for r in fixed['validation']} & {r['patient_id'] for r in fixed['test']}:
        raise ValueError('Validation and test patients overlap')
    if {r['pixel_sha256'] for r in fixed['validation']} & {r['pixel_sha256'] for r in fixed['test']}:
        raise ValueError('Validation and test pixels overlap')

    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'status.json', {'status': 'preparing'})
    train_paths = sorted((DATA / 'QaTa-COV19-v2/Train Set/Images').glob('*.png'))
    positive_paths = [p for p in train_paths if patient_id(p)]
    negative_paths = sorted((DATA / 'Control_Group/Control_Group_I').glob('*.png'))
    if len(train_paths) != 7145 or len(positive_paths) != 4194 or len(negative_paths) != 12544:
        raise ValueError('QaTa source inventory differs from the existing release')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        positive = list(pool.map(image_record, positive_paths))
        negative = list(pool.map(image_record, negative_paths))
    for label, records in [(1, positive), (0, negative)]:
        for row in records:
            # Normal file numbers are image IDs, never inferred patient IDs.
            if label == 0:
                row['patient_id'] = None
            row.update(label_vector=[label], labels=[label], label=repr([label]),
                       **{'class': repr(['covid'] if label else [])},
                       patient_id_known=row['patient_id'] is not None,
                       pixel_group_id='pixel:' + row['pixel_sha256'],
                       source_split='QaTa_v2_train_known_subject' if label else 'QaTa_Control_Group_I_documented_normal')
            row['group_id'] = row['patient_id'] or row['pixel_group_id']
    pos, neg, pool_audit = eligible_candidates(positive, negative, all_fixed)
    save_json(out / 'candidate_pool.json', {
        'positive_rows': pos, 'negative_rows': neg, 'audit': pool_audit,
        'positive_source': str(DATA / 'QaTa-COV19-v2/Train Set/Images'),
        'negative_source': str(DATA / 'Control_Group/Control_Group_I'),
        'negative_patient_ids_available': False,
        'unknown_patient_train_images_excluded': len(train_paths) - len(positive_paths),
        'metadata_policy': 'Use the existing QaTa release only; do not infer normal patient IDs from numeric filenames.',
    })
    protocol = {
        'status': 'frozen', 'version': 'qata-five-support-seeds-validation-threshold-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'qatacovid_seg', 'display_name': 'QaTa-COV19-v2', 'model': 'AlphaRAD',
        'class_names': ['covid'], 'eta': .3, 'crop_fraction': .8,
        'seeds': list(SEEDS), 'support_seeds': list(SEEDS), 'branches': list(BRANCHES),
        'evaluation_positive_only': True, 'source_protocol': ref(source),
        'candidate_pool': ref(out / 'candidate_pool.json'),
        'threshold_protocol': copy.deepcopy(original['threshold_protocol']),
        'primary_contrasts': [list(pair) for pair in PRIMARY_CONTRASTS],
        'secondary_contrasts': [list(pair) for pair in SECONDARY_CONTRASTS],
        'bootstrap': {'replicates': 2000, 'seed': 20261024, 'unit': 'patient',
                      'shared_across_seeds_and_branches': True, 'interval': 'percentile 95%',
                      'estimand': 'Mean of the five separately fitted and thresholded image-mean Dice metrics.',
                      'scope': 'Conditional on these five supports, fixed validation split and its frozen thresholds; does not include future support or threshold-selection uncertainty.',
                      'support_standard_deviation_ddof': 1, 'multiple_comparison_correction': False},
        'sampling_rule': 'For each independent seed 42-46: sort eligible positive patient IDs, sample five without replacement, select one image uniformly from each sorted patient image list; then sample five sorted negative exact-pixel representatives without replacement. Different seeds may overlap.',
        'deduplication_rule': 'Exclude all known validation/test patients. Exclude positive patients with any evaluation-pixel or cross-label pixel conflict; exclude conflicting negative pixels. Within each label retain the lexicographically first image per decoded-pixel hash.',
        'patient_id_rule': original['patient_id_rule'],
        'patient_identity_sources': original['patient_identity_sources'],
        'supervision_budget': {'query_support_images_per_seed': 10, 'positive_images_per_seed': 5,
                               'negative_images_per_seed': 5, 'validation_mask_images': 473,
                               'validation_patients': 200,
                               'positive_query_use': 'Only the same five positive images used by Q; the five negative labels are unused by this mechanism ablation.',
                               'validation_use': 'Only per-seed, per-branch scalar threshold selection; no query or model fitting.'},
        'limitations': [
            'Control_Group_I normals have no reliable patient IDs: exact-pixel separation is verified, but neither negative-support patient isolation nor five distinct negative patients can be claimed.',
            'Positive-only queries and Q share positive supports; Q additionally uses the five negative labels. This is an information ablation, not evidence that both branches use identical effective supervision.',
            'The fixed validation split uses 473 dense masks, beyond the ten image-level support labels per seed.',
            'The existing test set participated in earlier method development; this is not untouched external validation.',
            'All evaluation images are COVID-positive; healthy-image false-positive segmentation is not measured.',
            'Exact decoded-pixel checks do not detect all near-duplicate images.',
            'Support draws are independent seeded samples from one pool, and may overlap between seeds.',
        ],
        'source_files': [ref(Path(__file__)), ref(ROOT / 'data/alpharad/QaTaCOV19/readme.txt')],
    }
    protocol['threshold_protocol']['selection_data'] = 'validation only, independently per support seed and branch'
    protocol['threshold_protocol']['freeze_before_test_scoring'] = True
    protocol['threshold_protocol']['freeze_scope'] = 'All five seeds and all five branch thresholds saved before test scoring starts.'
    for split, rows in fixed.items():
        old = original[split]
        for kind in ['csv', 'rows']:
            target = out / (split + ('.csv' if kind == 'csv' else '_rows.json'))
            shutil.copyfile(old[kind + '_path'], target)
            if sha(target) != old[kind + '_sha256']:
                raise ValueError('Failed byte-identical frozen split copy')
        identity_path = out / (split + '_identities.json')
        save_json(identity_path, [{k: r[k] for k in IDENTITY_FIELDS} for r in rows])
        protocol[split + '_identities'] = ref(identity_path)
        protocol[split + '_identities_path'] = str(identity_path)
        protocol[split + '_identities_sha256'] = sha(identity_path)
        protocol[split] = {'csv_path': str(out / (split + '.csv')),
                           'csv_sha256': old['csv_sha256'], 'rows_path': str(out / (split + '_rows.json')),
                           'rows_sha256': old['rows_sha256'], 'rows_count': len(rows),
                           'patients_count': old['patients_count']}
        for suffix in ['csv_path', 'csv_sha256', 'rows_path', 'rows_sha256']:
            protocol[split + '_' + suffix.replace('csv_path', 'csv')] = protocol[split][suffix]
    supports, support_audits = {}, {}
    protocol['support_sets'] = {}
    for seed in SEEDS:
        rows = sample_support(pos, neg, seed)
        validate_support(rows, all_fixed)
        supports[seed] = rows
        rows_path, csv_path = out / f'support_seed_{seed}_rows.json', out / f'support_seed_{seed}.csv'
        save_json(rows_path, rows)
        with csv_path.open('w', newline='') as handle:
            keys = ['image', 'class', 'label', 'source_split', 'patient_id', 'pixel_sha256', 'group_id']
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: r[k] for k in keys} for r in rows)
        protocol['support_sets'][str(seed)] = {
            'seed': seed, 'rows_path': str(rows_path), 'rows_sha256': sha(rows_path), 'rows_count': 10,
            'csv_path': str(csv_path), 'csv_sha256': sha(csv_path), 'positive_count': 5, 'negative_count': 5,
            'known_patients_count': 5, 'unknown_patient_images': 5,
        }
        support_audits[str(seed)] = {'support_images': 10, 'positive_patients': 5,
                                    'negative_pixel_groups': 5, 'unknown_patient_images': 5,
                                    'known_patient_overlap_with_validation_or_test': [],
                                    'pixel_overlap_with_validation_or_test': [],
                                    'within_support_pixel_duplicates': [],
                                    'full_patient_isolation_verified': False}
    overlap = {}
    for i, left in enumerate(SEEDS):
        for right in SEEDS[i + 1:]:
            overlap[f'{left}_vs_{right}'] = {
                'images': sorted({r['image'] for r in supports[left]} & {r['image'] for r in supports[right]}),
                'known_patients': sorted({r['patient_id'] for r in supports[left] if r['patient_id']} &
                                         {r['patient_id'] for r in supports[right] if r['patient_id']}),
            }
    audit = {'source_protocol': ref(source), 'pool': pool_audit, 'seeds': support_audits,
             'between_seed_overlap': overlap,
             'validation_images': 473, 'validation_patients': 200, 'test_images': 2113, 'test_patients': 474,
             'validation_and_test_csv_and_rows_byte_identical_to_source': True,
             'validation_test_patient_overlap': [], 'validation_test_pixel_overlap': [],
             'frozen_evaluation_image_bytes_and_decoded_pixels_verified': True,
             'test_masks_opened_during_preparation': False,
             'decoded_pixel_hash_definition': "sha256(str(PIL_RGB.size).encode() + b'\\0RGB\\0' + PIL_RGB.tobytes())",
             'negative_patient_ids_available': False, 'full_patient_isolation_verified': False}
    save_json(out / 'PROTOCOL_AUDIT.json', audit)
    protocol['audit_path'], protocol['audit_sha256'] = str(out / 'PROTOCOL_AUDIT.json'), sha(out / 'PROTOCOL_AUDIT.json')
    save_json(out / 'protocol.json', protocol)
    save_json(out / 'catalog.json', {'status': 'frozen', 'seeds': list(SEEDS),
                                    'datasets': {'qatacovid_seg': ref(out / 'protocol.json')}})
    save_json(out / 'status.json', {'status': 'frozen', 'protocol_sha256': sha(out / 'protocol.json')})
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'protocol_v1')
    parser.add_argument('--source-protocol', type=Path, default=SOURCE)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    protocol = prepare(args.output, args.source_protocol, args.workers)
    print(json.dumps({'protocol': str(args.output.resolve() / 'protocol.json'),
                      'seeds': protocol['seeds'], 'validation_images': 473, 'test_images': 2113,
                      'negative_patient_ids_available': False}, indent=2))


if __name__ == '__main__':
    main()
