#!/usr/bin/env python3
"""Acquire five independent RSNA support draws using official NIH patient mapping.

Keep the existing test images/labels/order; exclude all NIH patients represented
in the original 5,337-image reference split from new support candidates.
Download only selected DICOMs, then execute AlphaRAD's original converter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import pydicom

import prepare_chest_support as common

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
SOURCE = HERE / 'source_data'
IMAGE_ROOT = MEDICAL / 'data/alpharad/RSNA/stability_train_support_20261024'
ZIP_URL = 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip'
ZIP_BYTES = 3978753654
SEEDS = [42, 43, 44, 45, 46]
CALCULATED = {'L_o8w': 0, 'L_yd0': 0, 'L_v8n': 1}
common.ZIP_URL, common.ZIP_BYTES = ZIP_URL, ZIP_BYTES
sys.path.insert(0, str(MEDICAL / 'AlphaRAD/local_setup'))
from prepare_storage_data import author_converter


def sample(rows, patient_all_labels, seed):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row['patient_id']].append(row)
    positive = sorted(p for p, items in by_patient.items() if any(r['labels'][0] == 1 for r in items))
    # Do not designate a patient negative if another exam is positive or unknown.
    negative = sorted(p for p in by_patient if patient_all_labels[p] == {0})
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    chosen = []
    for target, patients in ((1, positive), (0, negative)):
        for patient in rng.permutation(patients)[:5]:
            eligible = sorted((r for r in by_patient[patient] if r['labels'][0] == target), key=lambda r: r['image_id'])
            row = eligible[int(rng.integers(len(eligible)))]
            chosen.append({**row, 'anchor_class': 'Pneumonia' if target else 'negative'})
    if len(chosen) != 10 or len({r['patient_id'] for r in chosen}) != 10:
        raise ValueError('Require five positive and five negative distinct patients')
    return chosen, {'seed': seed, 'positive_counts': [5], 'negative_counts': [5],
                    'eligible_positive_patients': len(positive), 'eligible_strict_negative_patients': len(negative)}


def download_member(row, index, convert):
    info = index[row['sop_id'] + '.dcm']
    dcm = IMAGE_ROOT / 'dicom' / (row['image_id'] + '.dcm')
    if dcm.exists():
        raw = dcm.read_bytes()
    else:
        header = common.fetch(ZIP_URL, info['offset'], info['offset'] + 30)
        fields = struct.unpack('<4s5H3L2H', header)
        if fields[0] != b'PK\x03\x04' or fields[2] & 1:
            raise ValueError('Invalid/encrypted ZIP header')
        name_size, extra_size = fields[-2:]
        start = info['offset'] + 30
        packed = common.fetch(ZIP_URL, start, start + name_size + extra_size + info['compressed_bytes'])
        if packed[:name_size].decode() != info['archive_name']:
            raise ValueError('ZIP entry mismatch')
        packed = packed[name_size + extra_size:]
        raw = zlib.decompress(packed, -15) if info['compression'] == 8 else packed
        if len(raw) != info['bytes'] or zlib.crc32(raw) != info['crc32'] or raw[128:132] != b'DICM':
            raise ValueError('DICOM size/CRC/signature mismatch')
        common.frozen(dcm, raw)
    if len(raw) != info['bytes'] or zlib.crc32(raw) != info['crc32'] or raw[128:132] != b'DICM':
        raise ValueError('Cached DICOM identity mismatch')
    header = pydicom.dcmread(dcm, stop_before_pixels=True)
    expected = {'PatientID': row['image_id'], 'StudyInstanceUID': row['study_id'],
                'SeriesInstanceUID': row['series_id'], 'SOPInstanceUID': row['sop_id']}
    for key, value in expected.items():
        if str(getattr(header, key)) != value:
            raise ValueError(f'DICOM {key} disagrees with official mapping')
    png = IMAGE_ROOT / 'author_default512' / (row['image_id'] + '.png')
    png.parent.mkdir(parents=True, exist_ok=True)
    receipt = png.with_suffix('.json')
    source_sha = common.sha(dcm)
    if png.exists():
        meta = json.loads(receipt.read_text())
        if meta['source_sha256'] != source_sha or meta['png_sha256'] != common.sha(png):
            raise ValueError('Existing conversion receipt mismatch')
    else:
        convert(str(dcm), str(png), do_resize=True)
        common.write_json(receipt, {'source_sha256': source_sha, 'png_sha256': common.sha(png),
                                   'converter_sha256': common.sha(MEDICAL / 'AlphaRAD/cxrclip/data/dataformat_converter.py')})
    return row['image_id'], {**common.image_identity(png), 'source_image': str(dcm), 'source_sha256': source_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    mapping_path = SOURCE / 'pneumonia-challenge-dataset-mappings_2018.json'
    annotation_path = SOURCE / 'pneumonia-challenge-annotations-adjudicated-kaggle_2018.json'
    mappings = json.loads(mapping_path.read_text())
    annotations = json.loads(annotation_path.read_text())
    labels = defaultdict(set)
    for row in annotations['datasets'][0]['annotations']:
        if row['labelId'] in CALCULATED:
            labels[row['SOPInstanceUID']].add(CALCULATED[row['labelId']])
    if any(len(v) != 1 for v in labels.values()):
        raise ValueError('Inconsistent official calculated labels')
    lookup = {r['subset_img_id']: r for r in mappings}
    if len(lookup) != 30000:
        raise ValueError('Unexpected official mapping count')
    def patient(row):
        return 'NIH:' + row['img_id'].split('_')[0]
    # The official data card reports 12,274 original patients; UUIDs are per exam.
    if len({patient(r) for r in mappings}) != 12274:
        raise ValueError('NIH patient mapping does not match published cohort count')
    old_inventory = MEDICAL / 'RadZero_i2cl/local_assets/splits/native_public5_shot5_seed42/RSNA/inventory.csv'
    with old_inventory.open() as handle:
        old = list(csv.DictReader(handle))
    for row in old:
        if labels[lookup[row['patient_id']]['SOPInstanceUID']] != set(json.loads(row['labels_json'])):
            raise ValueError('Official calculated labels disagree with historical reference labels')
    excluded = {patient(lookup[r['patient_id']]) for r in old}
    patient_all_labels = defaultdict(set)
    rows = []
    for entry in sorted(mappings, key=lambda r: r['subset_img_id']):
        pid = patient(entry)
        values = labels.get(entry['SOPInstanceUID'], {-1})
        patient_all_labels[pid].update(values)
        if pid in excluded or values == {-1}:
            continue
        rows.append({'image_id': entry['subset_img_id'], 'patient_id': pid, 'group_id': pid,
                     'dicom_patient_id': entry['subset_img_id'], 'nih_image_id': entry['img_id'],
                     'study_id': entry['StudyInstanceUID'], 'series_id': entry['SeriesInstanceUID'],
                     'sop_id': entry['SOPInstanceUID'], 'labels': [next(iter(values))],
                     'source_split': 'official_labeled_non_reference_patients'})
    old_protocol_path = MEDICAL / 'experiments/fewshot_native_query/protocol_v1/rsna_pneumonia.json'
    old_protocol = json.loads(old_protocol_path.read_text())
    test_manifest = Path(old_protocol['test_manifest']['path'])
    test = json.loads(test_manifest.read_text())['rows']
    test_patients = {patient(lookup[Path(r['image']).stem]) for r in test}
    old_support_patients = {patient(lookup[Path(r['image']).stem]) for r in old_protocol['support_rows']}
    draws = {str(seed): sample(rows, patient_all_labels, seed) for seed in SEEDS}
    union = {r['image_id']: r for selected, _ in draws.values() for r in selected}
    frozen = {'version': 'rsna-official-nih-patient-separated-five-seeds-v1', 'seeds': SEEDS,
              'class_names': ['Pneumonia'], 'candidate_count': len(rows), 'candidate_rows': rows,
              'candidate_patient_count': len({r['patient_id'] for r in rows}),
              'test_manifest_sha256': common.sha(test_manifest), 'selected_union_count': len(union),
              'selection_uses_predictions': False, 'seed42_is_historical_support': False,
              'sampling': 'Sorted NIH patient pools; default_rng(SeedSequence([seed,0])); choose five distinct positive patients then five distinct strictly negative patients, one random eligible exam per patient. Negative patients have only zero labels across all their official RSNA exams, no positive or unknown exam. Exclude every NIH patient represented in all 5337 historical reference images before sampling.',
              'draws': {seed: {'support_rows': selected, 'sampling': meta} for seed, (selected, meta) in draws.items()},
              'sources': {str(p): common.sha(p) for p in (mapping_path, annotation_path, old_inventory, old_protocol_path)},
              'audit': {'original_reference_images': len(old), 'excluded_reference_patients': len(excluded),
                        'fixed_test_images': len(test), 'fixed_test_patients': len(test_patients),
                        'historical_support_patients': len(old_support_patients),
                        'historical_support_test_patient_overlap_count': len(old_support_patients & test_patients),
                        'historical_support_test_patient_overlap': sorted(old_support_patients & test_patients),
                        'official_labels_match_all_5337_historical_images': True,
                        'patient_id_definition': 'NIH original filename prefix, linked by official RSNA mapping. DICOM UUID is per examination, not patient.',
                        'patient_documentation': 'https://atlas.rsna.org/cards/74777c92-9b0b-4f06-a788-71a8a12c4e93',
                        'labels_source': 'Official Calculated label group: L_v8n=1; L_o8w/L_yd0=0; exclude missing calculated labels.'}}
    common.write_json(SOURCE / 'rsna_sampling_freeze.json', frozen)
    print(f'FROZEN RSNA candidates={len(rows)} patients={frozen["candidate_patient_count"]} union={len(union)} test_patients={len(test_patients)}', flush=True)
    index_path = SOURCE / 'rsna_zip_index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())['members']
    else:
        index = common.remote_index()
        common.write_json(index_path, {'url': ZIP_URL, 'archive_bytes': ZIP_BYTES, 'members': index})
    convert = author_converter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        images = dict(pool.map(lambda row: download_member(row, index, convert), union.values()))
        test_inventory = list(pool.map(lambda r: common.image_identity(r['image']), test))
    support_pixels = {r['pixel_sha256'] for r in images.values()}
    test_pixels = {r['pixel_sha256'] for r in test_inventory}
    if len(support_pixels) != len(images) or support_pixels & test_pixels:
        raise ValueError('Exact pixel duplicate within selected supports or across support/test; revise protocol before inference')
    final = {**frozen, 'seeds': {}, 'status': 'ready', 'test_image_inventory': test_inventory,
             'image_acquisition_scope': 'Only the selected support union is downloaded; candidate pool is complete metadata, not a full image download.'}
    final.pop('draws')
    for seed, (selected, meta) in draws.items():
        support = [{**row, **images[row['image_id']]} for row in selected]
        if {r['patient_id'] for r in support} & test_patients:
            raise ValueError('Support/test NIH patient overlap')
        final['seeds'][seed] = {'support_rows': support, 'sampling': meta}
    final['audit'].update({'support_test_patient_overlap': 0, 'support_test_pixel_overlap': 0,
                           'support_union_unique_pixels': len(support_pixels), 'fixed_test_unique_pixels': len(test_pixels),
                           'all_selected_dicom_headers_and_crc_verified': True})
    common.write_json(SOURCE / 'rsna_pool.json', final)
    print('READY', SOURCE / 'rsna_pool.json', flush=True)


if __name__ == '__main__':
    main()
