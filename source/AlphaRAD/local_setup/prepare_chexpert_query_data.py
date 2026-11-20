#!/usr/bin/env python3
"""Freeze CheXpert full-test and patient-disjoint five-shot comparison manifests.

Uses the author's image order and 12-class constant. Sampling uses labels and
image identity only, never model predictions. Existing data/results are not
modified. Because only the released test images are available locally, support
patients (including every view) are excluded from the matched evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random

from prepare_storage_data import ROOT, ALPHA, atomic_json, read_csv, save_csv, sha, source_file


def author_classes(path):
    tree = ast.parse(path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CHEXPERT' for t in n.targets))
    return list(ast.literal_eval(assignment.value))


def decoded_identity(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f'Unreadable grayscale image: {path}')
    return {'pixel_sha256': hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest(),
            'decoded_shape': list(image.shape), 'byte_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ALPHA / 'CheXpert')
    parser.add_argument('--author-csv', type=Path, default=ROOT / 'configs/data_csv/chexpert.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.shots != 5:
        raise ValueError('This frozen protocol requires exactly five anchor images per class.')
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Use a new output directory: {args.output}')
    constants_path = ROOT / 'cxrclip/prompt/constants.py'
    classes = author_classes(constants_path)
    label_path = args.data_root / 'test_labels.csv'
    source_rows = read_csv(label_path)
    label_columns = [name for name in source_rows[0] if name != 'Path']
    source_by_path = {row['Path']: row for row in source_rows}
    if len(source_by_path) != len(source_rows):
        raise ValueError('Duplicate paths in released labels.')
    original_rows = read_csv(args.author_csv)
    full = []
    for row in original_rows:
        path = str(row['image'])
        relative = 'test/' + path[path.index('patient'):]
        raw = source_by_path[relative]
        raw_y = [float(raw[c]) for c in label_columns]
        if any(y not in (0., 1.) for y in raw_y):
            raise ValueError(f'Unknown or uncertain label: {relative}')
        if ast.literal_eval(row['label']) != raw_y:
            raise ValueError(f'Author/source 14-label mismatch: {relative}')
        old_names = ast.literal_eval(row['class'])
        if set(old_names) != {c for c, y in zip(label_columns, raw_y) if y}:
            raise ValueError(f'Author/source disease-name mismatch: {relative}')
        image = args.data_root / relative
        if not image.is_file():
            raise FileNotFoundError(image)
        patient, study, filename = relative.split('/')[1:]
        y = [int(float(raw[c])) for c in classes]
        full.append({'image': str(image), 'label': repr(y),
                     'class': repr([c for c, value in zip(classes, y) if value]),
                     'patient_id': patient, 'study_id': patient + '/' + study,
                     'view': 'frontal' if 'frontal' in filename else 'lateral' if 'lateral' in filename else 'unknown'})
    if len(full) != 668 or len({r['image'] for r in full}) != len(full):
        raise ValueError('Expected all 668 unique images in the released author ordering.')
    if set(source_by_path) != {str(Path(r['image']).relative_to(args.data_root)) for r in full}:
        raise ValueError('Author image list does not equal the released test-label image list.')
    import cv2
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        identities = list(pool.map(decoded_identity, [r['image'] for r in full]))
    for row, identity in zip(full, identities):
        row['pixel_sha256'] = identity['pixel_sha256']
    ys = [ast.literal_eval(r['label']) for r in full]
    patients = [r['patient_id'] for r in full]
    patient_indices, pixel_patients = defaultdict(list), defaultdict(set)
    for i, row in enumerate(full):
        patient_indices[row['patient_id']].append(i)
        pixel_patients[row['pixel_sha256']].add(row['patient_id'])
    if any(len({tuple(ys[i]) for i in indices}) != 1 for indices in patient_indices.values()):
        raise ValueError('Expected consistent study labels across same-patient views.')

    # Merge patient groups connected by an exactly identical decoded image.
    parent = {p: p for p in patient_indices}
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    for members in pixel_patients.values():
        members = sorted(members)
        for p in members[1:]:
            parent[find(p)] = find(members[0])
    groups = [find(p) for p in patients]
    minimum_positive = [1 if c == 'Fracture' else 2 for c in classes]
    order = sorted(range(len(classes)), key=lambda c: (sum(y[c] for y in ys), c))
    rng = random.Random(args.seed)
    selected, selected_pixels, blocked_groups, support = set(), set(), set(), []
    decisions = []
    for c in order:
        pool = [i for i, y in enumerate(ys) if y[c]]
        rng.shuffle(pool)
        count = 0
        skipped = Counter()
        for i in pool:
            if i in selected or full[i]['pixel_sha256'] in selected_pixels:
                skipped['already_selected_image_or_pixels'] += 1
                continue
            blocked = blocked_groups | {groups[i]}
            holdout = [j for j, group in enumerate(groups) if group not in blocked]
            feasible = all(
                len({patients[j] for j in holdout if ys[j][d]}) >= minimum_positive[d]
                and len({patients[j] for j in holdout if not ys[j][d]}) >= 2
                for d in range(len(classes)))
            if not feasible:
                skipped['would_remove_required_holdout_label_groups'] += 1
                continue
            selected.add(i)
            selected_pixels.add(full[i]['pixel_sha256'])
            blocked_groups = blocked
            support.append({**full[i], 'anchor_class': classes[c], 'source_split': 'released_test_support_holdout'})
            count += 1
            if count == args.shots:
                break
        decisions.append({'class': classes[c], 'anchors_selected': count, 'skipped': dict(skipped)})
        if count != args.shots:
            raise ValueError(f'Cannot select {args.shots} distinct anchors for {classes[c]} while keeping all test classes evaluable; selected {count}.')
    evaluated = [r for i, r in enumerate(full) if groups[i] not in blocked_groups]
    excluded = [{**r, 'exclusion_reason': 'support_image' if i in selected else 'support_patient_or_duplicate_connected_patient'}
                for i, r in enumerate(full) if groups[i] in blocked_groups]
    if len(support) != len(classes) * args.shots or len({r['image'] for r in support}) != len(support):
        raise ValueError('Support anchors are not distinct or the count is wrong.')
    if {r['patient_id'] for r in support} & {r['patient_id'] for r in evaluated}:
        raise ValueError('Support/evaluation patient overlap.')
    if {r['pixel_sha256'] for r in support} & {r['pixel_sha256'] for r in evaluated}:
        raise ValueError('Support/evaluation decoded-pixel overlap.')
    def counts(rows):
        yy = [ast.literal_eval(r['label']) for r in rows]
        positive = [sum(y[c] for y in yy) for c in range(len(classes))]
        return {'positive_count': positive, 'negative_count': [len(rows) - n for n in positive],
                'positive_patient_count': [len({r['patient_id'] for r, y in zip(rows, yy) if y[c]}) for c in range(len(classes))],
                'patient_count': len({r['patient_id'] for r in rows}), 'image_count': len(rows),
                'view_count': dict(Counter(r['view'] for r in rows))}
    full_counts, support_counts, test_counts = map(counts, (full, support, evaluated))
    if min(support_counts['positive_count'] + support_counts['negative_count'] + test_counts['positive_count'] + test_counts['negative_count']) < 1:
        raise ValueError('Support or matched test contains a single-outcome class.')
    args.output.mkdir(parents=True, exist_ok=False)
    full_path = args.output / 'chexpert_full.csv'
    support_path = args.output / 'chexpert_support.csv'
    eval_path = args.output / 'chexpert_evaluation.csv'
    excluded_path = args.output / 'chexpert_excluded.csv'
    for path, rows in ((full_path, full), (support_path, support), (eval_path, evaluated), (excluded_path, excluded)):
        save_csv(path, rows, list(rows[0]))
    limitations = [
        'Only the released 668-image test set is locally complete; this is a new labeled few-shot patient-holdout protocol.',
        'Fracture has only 6 images from 5 positive patients. Five unique positive support anchors necessarily leave at most 1 positive test patient; its metric is highly unstable.',
        'Multiple views from a support patient may serve as separate anchors. Every image of that patient is excluded from matched evaluation.',
        'Exact decoded grayscale duplicates are excluded across support/evaluation; near-duplicates are not searched.',
        'The full 668-image original zero-shot result and reduced matched comparison are separate evaluation populations and must not be subtracted from one another.',
    ]
    audit_path = args.output / 'chexpert_split_audit.json'
    audit = {'status': 'complete', 'class_names': classes, 'seed': args.seed,
             'shots_per_anchor_class': args.shots, 'anchor_class_counts': dict(Counter(r['anchor_class'] for r in support)),
             'sampling_order': [classes[c] for c in order],
             'sampling_algorithm': 'Seeded shuffle in author image order; rare classes first; unique image/pixel anchors; reject choices removing required patient-level holdout outcomes.',
             'minimum_holdout_positive_patients': dict(zip(classes, minimum_positive)),
             'sampling_decisions': decisions, 'model_predictions_used_for_sampling': False,
             'full': full_counts, 'support': support_counts, 'evaluation': test_counts,
             'support_count': len(support), 'evaluation_count': len(evaluated), 'original_evaluation_count': len(full),
             'removed_from_original_evaluation': len(excluded), 'support_patient_ids': sorted({r['patient_id'] for r in support}),
             'patient_overlap_count': 0, 'study_overlap_count': 0, 'pixel_overlap_count': 0,
             'cross_patient_duplicate_sets': [sorted(v) for v in pixel_patients.values() if len(v) > 1],
             'image_inventory': [{**r, **identity} for r, identity in zip(full, identities)],
             'excluded_csv_path': str(excluded_path), 'excluded_csv_sha256': sha(excluded_path),
             'limitations': limitations,
             'source_files': [source_file(p) for p in (Path(__file__), args.author_csv, label_path, constants_path)]}
    atomic_json(audit_path, audit)
    base = {'task': 'classification', 'display_name': 'CheXpert', 'class_names': classes,
            'csv_path': str(full_path), 'csv_sha256': sha(full_path), 'expected_count': len(full),
            'actual_count': len(full), 'unique_images': len(full), 'positive_count': full_counts['positive_count'],
            'scope': 'complete released author test list; original model only',
            'notes': ['Author image order and all frontal/lateral views retained.',
                      'Author CHEXPERT constant evaluates 12 classes: Pleural Other and Support Devices are omitted from original 14-label CSV vectors.'],
            'sources': audit['source_files'] + [source_file(audit_path)]}
    created = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output / 'full_catalog.json', {'status': 'complete', 'created_at_utc': created, 'datasets': {'chexpert': base}})
    adapted = {**base, 'csv_path': str(eval_path), 'csv_sha256': sha(eval_path),
               'expected_count': len(evaluated), 'actual_count': len(evaluated), 'unique_images': len(evaluated),
               'positive_count': test_counts['positive_count'], 'support_csv_path': str(support_path),
               'support_csv_sha256': sha(support_path), 'support_count': len(support),
               'support_positive_count': support_counts['positive_count'], 'support_negative_count': support_counts['negative_count'],
               'original_csv_path': str(full_path), 'original_csv_sha256': sha(full_path),
               'original_evaluation_count': len(full), 'removed_from_original_evaluation': len(excluded),
               'split_audit_path': str(audit_path), 'split_audit_sha256': sha(audit_path),
               'scope': 'matched baseline/refined evaluation after support-patient and duplicate exclusion',
               'notes': base['notes'] + limitations}
    atomic_json(args.output / 'catalog.json', {'status': 'complete', 'created_at_utc': created,
                'protocol': {'seed': args.seed, 'eta': 0.3, 'no_gradient_training': True,
                             'support_only_for_query_direction': True,
                             'baseline_and_refined_use_identical_evaluation_images': True},
                'datasets': {'chexpert': adapted}})
    print(f'READY CheXpert: full={len(full)}, support={len(support)} images/{support_counts["patient_count"]} patients, matched_test={len(evaluated)} images/{test_counts["patient_count"]} patients, excluded={len(excluded)}', flush=True)
    print(f'Catalog: {args.output / "catalog.json"}', flush=True)
    print(f'Full baseline catalog: {args.output / "full_catalog.json"}', flush=True)


if __name__ == '__main__':
    main()
