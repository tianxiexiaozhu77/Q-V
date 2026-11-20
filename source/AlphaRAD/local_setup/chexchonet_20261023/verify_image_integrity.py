#!/usr/bin/env python3
"""Read-only official image SHA256, decode, and cross-partition duplicate audit."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent
DATA = OUT.parents[2] / 'data/alpharad/chexchonet-custom-test-7494'


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    full = rows(OUT / 'full_evaluation.csv')
    expected = {}
    for line in (DATA / 'test_SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[str((DATA / name.lstrip('*')).resolve())] = digest
    assert len(expected) == len(full) == 7494
    support = {r['image'] for r in rows(OUT / 'support_union.csv')}
    support_patients = {r['patient_id'] for r in rows(OUT / 'support_union.csv')}
    evaluation = {r['image'] for r in rows(OUT / 'evaluation.csv')}

    def check(row):
        data = Path(row['image']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected[row['image']]:
            raise ValueError('Official checksum mismatch')
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            rgb = image.convert('RGB')
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            return {'image': row['image'], 'patient_id': row['patient_id'], 'size': list(rgb.size),
                    'sha256': digest, 'pixel_sha256': pixels, 'bytes': len(data)}

    inventory = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, record in enumerate(pool.map(check, full), 1):
            inventory.append(record)
            if i % 1000 == 0:
                print(f'Verified {i}/7494', flush=True)

    by_pixel = defaultdict(list)
    for record in inventory:
        by_pixel[record['pixel_sha256']].append(record)
    duplicates = [rs for rs in by_pixel.values() if len(rs) > 1]
    cross_patient = [rs for rs in duplicates if len({r['patient_id'] for r in rs}) > 1]
    cross_partition = [rs for rs in duplicates
                       if any(r['patient_id'] in support_patients for r in rs)
                       and any(r['image'] in evaluation for r in rs)]
    result = {
        'status': 'passed' if not cross_partition else 'failed_support_test_duplicate',
        'verified_images': len(inventory), 'verified_bytes': sum(r['bytes'] for r in inventory),
        'all_official_sha256_match': True, 'all_images_decode': True,
        'decoded_size_counts': dict(Counter(str(r['size']) for r in inventory)),
        'duplicate_pixel_groups': len(duplicates), 'cross_patient_duplicate_pixel_groups': len(cross_patient),
        'support_patient_test_duplicate_pixel_groups': len(cross_partition),
        'patient_overlap_count': len(support_patients & {r['patient_id'] for r in full if r['image'] in evaluation}),
        'composite_matches_source_boolean_or_count': sum(int(r['composite_slvh_dlv']) == (int(r['slvh']) or int(r['dlv'])) for r in full),
        'label_note': 'Composite labels were loaded from source unchanged; OR check is descriptive only.',
        'inventory': inventory,
        'cross_patient_duplicate_groups': cross_patient,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [DATA / 'test_SHA256SUMS.txt', OUT / 'catalog.json', Path(__file__)]},
    }
    (OUT / 'image_integrity_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('inventory', 'cross_patient_duplicate_groups', 'source_sha256')}, indent=2))
    if cross_partition:
        raise RuntimeError('Cross-partition decoded duplicate requires protocol repair before paired evaluation')


if __name__ == '__main__':
    main()
