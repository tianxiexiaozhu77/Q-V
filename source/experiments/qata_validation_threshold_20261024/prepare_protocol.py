#!/usr/bin/env python3
"""Freeze a patient-separated QaTa validation split without model predictions.

Keep the historical ten supports and the 2,113 test rows exactly unchanged.
Unknown-subject training images are deliberately not used for validation.
Only NumPy and Pillow are needed; no model, GPU or prediction is loaded.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'data/alpharad/QaTaCOV19/QaTa-COV19/QaTa-COV19-v2'
OLD = ROOT / 'AlphaRAD/local_setup/query_refinement_data_20261021'
SUPPORT = OLD / 'qatacovid_seg_support.csv'
TEST = OLD / 'qatacovid_seg_evaluation.csv'
AUTHOR_CSV = ROOT / 'AlphaRAD/configs/data_csv/qatacov19.csv'
UTILS = ROOT / 'AlphaRAD/cxrclip/evaluator_utils.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def patient_id(path):
    m = re.match(r'^(sub-S\d+)_', Path(path).name)
    return m.group(1) if m else None


def image_record(path):
    path = Path(path)
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + b'\0RGB\0' + rgb.tobytes()).hexdigest()
        size = list(rgb.size)
    return {'image': str(path), 'image_path': str(path), 'image_id': path.name,
            'patient_id': patient_id(path), 'group_id': patient_id(path),
            'image_size_wh': size, 'image_sha256': sha(path), 'pixel_sha256': pixel}


def load_native_rle():
    """Execute only the author's two pure NumPy RLE functions, verbatim."""
    tree = ast.parse(UTILS.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'mask2rle', 'rle2mask'}]
    assert len(nodes) == 2
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UTILS), 'exec'), ns)
    return ns['mask2rle'], ns['rle2mask']


def segmentation_row(record, source_row, encode, decode):
    row = dict(record)
    image = Path(row['image'])
    mask_path = image.parent.parent / 'Ground-truths' / ('mask_' + image.name)
    with Image.open(mask_path) as handle:
        mask = np.asarray(handle.convert('L'))
    if mask.shape != tuple(reversed(row['image_size_wh'])):
        raise ValueError(f'Mask/image shape mismatch: {image}')
    if not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError(f'Non-binary source mask: {mask_path}')
    binary = (mask > 0).astype(np.uint8)
    if source_row is None:
        rle = encode(binary) if binary.any() else '-1'
        source_row = {'image': str(image), 'label': '1.0', 'class': 'covid',
                      'EncodedPixels': repr([rle])}
    encoded = ast.literal_eval(source_row['EncodedPixels'])
    restored = np.zeros_like(binary, dtype=float)
    if encoded[0].strip() != '-1':
        for part in encoded:
            restored += decode(part.strip(), mask.shape[1], mask.shape[0])
    if not np.array_equal(binary, restored > 0):
        raise ValueError(f'Native RLE differs from source mask: {image}')
    row.update(source_row)
    row.update(mask=str(mask_path), mask_path=str(mask_path),
               mask_sha256=sha(mask_path), mask_positive_pixels=int(binary.sum()),
               label_vector=[1], native_rle_equals_source_mask=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'protocol_v1')
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--validation-patients', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.validation_patients < 1:
        raise ValueError('validation-patients must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'status.json', {'status': 'preparing'})

    train_paths = sorted((DATA / 'Train Set/Images').glob('*.png'))
    test_paths = sorted((DATA / 'Test Set/Images').glob('*.png'))
    assert len(train_paths) == 7145 and len(test_paths) == 2113
    native_test = read_csv(TEST)
    source_support = read_csv(SUPPORT)
    author_test = read_csv(AUTHOR_CSV)
    assert len(native_test) == 2113 and len(source_support) == 10
    assert set(r['image'] for r in native_test) == set(map(str, test_paths))
    for local, author in zip(native_test, author_test, strict=True):
        assert Path(local['image']).name == Path(author['image']).name
        assert all(local[k] == author[k] for k in ('class', 'label', 'EncodedPixels'))
    known_train = [p for p in train_paths if patient_id(p)]
    unknown_train = [p for p in train_paths if not patient_id(p)]
    all_paths = sorted(set(known_train + test_paths + [Path(r['image']) for r in source_support]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = {r['image']: r for r in pool.map(image_record, all_paths)}
    train_patients = {patient_id(p) for p in known_train}
    test_patients = {patient_id(p) for p in test_paths}
    support_patients = {patient_id(r['image']) for r in source_support if patient_id(r['image'])}
    assert None not in test_patients
    assert not support_patients & test_patients
    support_hashes = {records[r['image']]['pixel_sha256'] for r in source_support}
    test_hashes = {records[str(p)]['pixel_sha256'] for p in test_paths}
    assert not support_hashes & test_hashes, 'Historical support/test exact pixel overlap'
    excluded_pixel_patients = {patient_id(p) for p in known_train
                               if records[str(p)]['pixel_sha256'] in support_hashes | test_hashes}
    eligible = sorted(train_patients - support_patients - test_patients - excluded_pixel_patients)
    n_validation = args.validation_patients
    if n_validation >= len(eligible):
        raise ValueError('Validation must leave some eligible training patients unused')
    chosen = sorted(np.random.default_rng(args.seed).choice(eligible, n_validation, replace=False).tolist())
    chosen_set = set(chosen)
    val_paths = [p for p in known_train if patient_id(p) in chosen_set]
    encode, decode = load_native_rle()
    validation = [segmentation_row(records[str(p)], None, encode, decode) for p in val_paths]
    test = [segmentation_row(records[r['image']], r, encode, decode) for r in native_test]
    support = []
    for row in source_support:
        obj = dict(records[row['image']])
        obj.update({k: v for k, v in row.items() if k not in ('patient_id', 'pixel_sha256')})
        obj['label_vector'] = ast.literal_eval(row['label'])
        obj['patient_id_known'] = obj['patient_id'] is not None
        obj['group_id'] = obj['patient_id'] or 'unknown_subject_image:' + obj['image_id']
        support.append(obj)
    assert sum(r['label_vector'][0] for r in support) == 5
    val_hashes = {r['pixel_sha256'] for r in validation}
    assert not val_hashes & (test_hashes | support_hashes)
    assert not chosen_set & (test_patients | support_patients)

    # Save split assignment before any model prediction or threshold comparison.
    save_json(out / 'patient_assignment.json', {
        'eligible_train_patients': eligible, 'validation_patients': chosen,
        'unused_train_patients': sorted(set(eligible) - chosen_set),
        'test_patients': sorted(test_patients), 'support_known_patients': sorted(support_patients),
        'excluded_pixel_matching_patients': sorted(excluded_pixel_patients),
        'unknown_patient_train_images_not_used': [str(p) for p in unknown_train]})
    with (out / 'validation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'label', 'class', 'EncodedPixels'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in validation)
    shutil.copyfile(TEST, out / 'test.csv')
    shutil.copyfile(SUPPORT, out / 'support.csv')
    assert sha(out / 'test.csv') == sha(TEST) and sha(out / 'support.csv') == sha(SUPPORT)
    for split, rows in [('validation', validation), ('test', test), ('support', support)]:
        save_json(out / (split + '_rows.json'), rows)
    save_json(out / 'test_identities.json', [
        {k: r[k] for k in ('image', 'image_path', 'image_id', 'patient_id', 'group_id',
                           'image_size_wh', 'image_sha256', 'pixel_sha256')}
        for r in test])

    duplicates = {}
    for split, rows in [('validation', validation), ('test', test), ('support', support)]:
        by_hash = defaultdict(list)
        for row in rows:
            by_hash[row['pixel_sha256']].append({'image_id': row['image_id'], 'patient_id': row['patient_id']})
        duplicates[split] = [v for v in by_hash.values() if len(v) > 1]
    audit = {
        'raw_train_images': len(train_paths), 'raw_train_known_patient_images': len(known_train),
        'raw_train_known_patients': len(train_patients), 'raw_train_unknown_patient_images': len(unknown_train),
        'raw_train_test_known_patient_overlap': sorted(train_patients & test_patients),
        'support_known_patients': len(support_patients), 'support_unknown_patient_images': sum(not r['patient_id_known'] for r in support),
        'eligible_train_patients': len(eligible), 'validation_patients': len(chosen), 'validation_images': len(validation),
        'test_patients': len(test_patients), 'test_images': len(test),
        'validation_test_patient_overlap': [], 'validation_support_known_patient_overlap': [],
        'support_test_known_patient_overlap': [], 'validation_test_pixel_overlap': [],
        'validation_support_pixel_overlap': [], 'support_test_pixel_overlap': [],
        'excluded_pixel_matching_train_patients': sorted(excluded_pixel_patients),
        'within_split_pixel_duplicates': duplicates,
        'validation_empty_masks': sum(r['mask_positive_pixels'] == 0 for r in validation),
        'test_empty_masks': sum(r['mask_positive_pixels'] == 0 for r in test),
        'test_native_rle_exact_source_mask_matches': len(test),
        'validation_native_rle_roundtrip_exact_source_mask_matches': len(validation),
        'test_order_labels_rle_identical_to_author_csv': True,
        'test_csv_byte_identical_to_historical_evaluation': True,
        'support_csv_byte_identical_to_historical_support': True,
        'decoded_pixel_hash_definition': "sha256(str(PIL_RGB.size).encode() + b'\\0RGB\\0' + PIL_RGB.tobytes())",
        'image_size_counts': {s: dict(Counter(str(r['image_size_wh']) for r in rows)) for s, rows in [('validation', validation), ('test', test)]},
    }
    save_json(out / 'SPLIT_AUDIT.json', audit)
    limits = [
        'Validation uses only the explicitly identifiable BIMCV sub-S subjects from the released training partition; 2,951 covid_*.png training images without reliable subject mapping are unused.',
        'Five historical normal support images have no reliable patient ID; exact-pixel separation is verified but their patient-level separation cannot be independently verified.',
        'The 2,113-image test set was used in earlier method development; validation-selected thresholds remove current test-threshold optimization but do not make this an untouched external test set.',
        'The native positive-only COVID segmentation protocol does not measure healthy-image false positives.',
    ]
    protocol = {
        'status': 'frozen', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': 'qatacovid_seg', 'display_name': 'QaTa-COV19-v2', 'model': 'AlphaRAD',
        'class_names': ['covid'], 'eta': .3, 'crop_fraction': .8,
        'branches': ['original', 'query_refined', 'tta_center_original', 'tta_center_refined'],
        'evaluation_positive_only': True, 'seed': args.seed, 'validation_patients_requested': args.validation_patients,
        'preexisting_plan': {
            'path': str(ROOT / 'baselines/paper_draft_zh/RESEARCH_VALIDATION_PLAN.txt'),
            'sha256_at_freeze': sha(ROOT / 'baselines/paper_draft_zh/RESEARCH_VALIDATION_PLAN.txt'),
            'specification': 'Section six prespecified 200 training patients, seed 20261024, validation threshold selection, and fixed 0.5 secondary analysis.',
            'pre_inference_correction': 'An initial 20-percent (331-patient) draft completed data integrity checks only. It was superseded before any model inference or threshold selection to honor the preexisting 200-patient plan.',
        },
        'supervision_budget': {
            'query_support_images': 10, 'query_support_positive_images': 5, 'query_support_negative_images': 5,
            'validation_mask_images': len(validation), 'validation_patients': len(chosen),
            'validation_use': 'Dense masks used only to select four frozen branch-specific scalar thresholds; not used to fit queries or model weights.',
            'claim': 'The query refinement uses ten image labels; the complete calibrated segmentation pipeline additionally uses the listed validation masks.',
        },
        'sampling_rule': 'Sort eligible sub-S IDs; numpy.default_rng(seed).choice(ids, validation_patients, replace=False); retain all training images of chosen patients sorted by filename.',
        'patient_id_rule': 'BIMCV/MIDS filename leading sub-S<number>; ses-E is examination, not patient.',
        'patient_identity_sources': [
            'https://github.com/BIMCV-CSUSP/BIMCV-COVID-19',
            'https://arxiv.org/html/2010.00434v1',
            'https://bimcv.cipf.es/bimcv-projects/mids/',
        ],
        'threshold_protocol': {
            'selection_data': 'validation only, independently per branch',
            'grid': np.arange(0, 1.01, .01).tolist(),
            'grid_comparison': 'strict probability > float32(threshold)',
            'normalization': 'Per-image per-query minmax over all 1370 tokens including CLS, denominator max-min+1e-8; then drop CLS, reshape37x37, bilinear upsample224x224 align_corners=False, sigmoid.',
            'mean_dice': 'Foreground Dice over positive images, samplewise image-equal mean; native empty-mask rules preserved.',
            'tie_break': 'smallest threshold among exact maximum validation mean Dice ties',
            'freeze_before_test_scoring': True,
            'test_role': 'evaluate once at the frozen branch-specific validation threshold',
            'secondary_fixed_threshold': .5,
            'secondary_fixed_threshold_role': 'Prespecified secondary result without validation-mask threshold calibration; no threshold chosen from test performance.',
        },
        'limitations': limits,
        'source_files': [{'path': str(p), 'sha256': sha(p)} for p in [Path(__file__).resolve(), SUPPORT, TEST, AUTHOR_CSV, UTILS, ROOT/'data/alpharad/QaTaCOV19/readme.txt']],
        'audit_path': str(out / 'SPLIT_AUDIT.json'), 'audit_sha256': sha(out / 'SPLIT_AUDIT.json'),
        'patient_assignment_path': str(out / 'patient_assignment.json'), 'patient_assignment_sha256': sha(out / 'patient_assignment.json'),
        'test_identities_path': str(out / 'test_identities.json'),
        'test_identities_sha256': sha(out / 'test_identities.json'),
    }
    for split, rows in [('validation', validation), ('test', test), ('support', support)]:
        csv_path = out / (split + '.csv')
        rows_path = out / (split + '_rows.json')
        protocol[split] = {'csv_path': str(csv_path), 'csv_sha256': sha(csv_path),
                           'rows_path': str(rows_path), 'rows_sha256': sha(rows_path),
                           'rows_count': len(rows), 'patients_count': len({r['patient_id'] for r in rows if r['patient_id']})}
        protocol[split + '_csv'] = str(csv_path)
        protocol[split + '_csv_sha256'] = sha(csv_path)
        protocol[split + '_rows_path'] = str(rows_path)
        protocol[split + '_rows_sha256'] = sha(rows_path)
    save_json(out / 'protocol.json', protocol)
    shutil.copyfile(out / 'protocol.json', out / 'catalog.json')
    save_json(out / 'status.json', {'status': 'frozen', 'protocol_sha256': sha(out / 'protocol.json')})
    print(json.dumps({k: audit[k] for k in ['raw_train_images','raw_train_known_patient_images','raw_train_unknown_patient_images','eligible_train_patients','validation_patients','validation_images','test_patients','test_images','validation_empty_masks','test_empty_masks']}, indent=2))
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
