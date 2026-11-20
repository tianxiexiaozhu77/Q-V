#!/usr/bin/env python3
"""Frozen eight-baseline CheXchoNet transfer with exact, pre-existing supports.

Outputs all four arms on the same patients. The original arm is checked against
the native adapter, without requiring an unrelated historical prediction file.
Each architecture retains the score-domain fusion used in the previous eight
baseline campaign. No test labels are used for fitting or parameter selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from evaluate_baseline_buffered import sha, write, now, load_adapter, classification_metrics
from evaluate_query_transfer import load_query, paired, check_native

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ('original', 'query_refined', 'tta_center_original', 'tta_center_refined')
TASKS = ['chexchonet_slvh', 'chexchonet_dlv', 'chexchonet_composite_slvh_dlv']
JOINT = {'BiomedCLIP', 'BioViL-T', 'CheXzero', 'MaCo'}


def validate_inputs(catalog_path):
    catalog = json.loads(catalog_path.read_text())
    inputs = []
    for name in TASKS:
        entry = catalog['datasets'][name]
        for key, hash_key in [('manifest', 'sha256'), ('support_manifest', 'support_manifest_sha256')]:
            if sha(entry[key]) != entry[hash_key]:
                raise ValueError('Changed fixed manifest: ' + entry[key])
        data = json.loads(Path(entry['manifest']).read_text())
        supports = json.loads(Path(entry['support_manifest']).read_text())
        rows, support = data['rows'], supports['support_rows']
        if data['task'] != 'classification' or len(data['class_names']) != 1:
            raise ValueError('Each task must be binary classification')
        if len(rows) != 7407 or len(support) != 10 or sum(r['labels'][0] for r in support) != 5:
            raise ValueError('Frozen support/test counts changed')
        if len({r['patient_id'] for r in support}) != 10:
            raise ValueError('Supports must have distinct patients')
        if {r['patient_id'] for r in support} & {r['patient_id'] for r in rows}:
            raise ValueError('Support/test patient overlap')
        if len({r['image'] for r in rows}) != 7407:
            raise ValueError('Duplicate test path')
        for path, digest in supports['support_sources'].items():
            if sha(path) != digest:
                raise ValueError('Support source drift: ' + path)
        for row in support:
            if sha(row['image']) != row['byte_sha256']:
                raise ValueError('Support image changed')
        if inputs and [(r['image'], r['patient_id']) for r in rows] != [(r['image'], r['patient_id']) for r in inputs[0]['rows']]:
            raise ValueError('All tasks must use identical ordered test patients/images')
        inputs.append(dict(name=name, entry=entry, data=data, supports=supports,
                           support_rows=support, rows=rows, classes=data['class_names']))
    union = {r['patient_id'] for t in inputs for r in t['support_rows']}
    if len(union) != 30 or union & {r['patient_id'] for r in inputs[0]['rows']}:
        raise ValueError('Invalid shared patient holdout')
    return catalog, inputs


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    catalog, inputs = validate_inputs(args.catalog)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        view_weight = float(os.environ.get('QUERY_REFINEMENT_VIEW_WEIGHT', '0.5'))
    except ValueError as exc:
        raise ValueError('QUERY_REFINEMENT_VIEW_WEIGHT must be numeric') from exc
    if not np.isfinite(view_weight) or not 0.0 <= view_weight <= 1.0:
        raise ValueError('QUERY_REFINEMENT_VIEW_WEIGHT must lie in [0, 1]')
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned GPU unavailable')
    source_paths = [Path(__file__), Path(__file__).with_name('evaluate_query_transfer.py'),
        Path(__file__).with_name('query_refinement_common.py'), Path(__file__).with_name('evaluate_baseline_buffered.py'),
        ROOT / args.model / 'local_setup/query_refinement_adapter.py',
        ROOT / args.model / 'local_setup/adapter.py', args.model_root / 'local_setup/adapter.py',
        args.model_root / 'local_setup/manifest.json', args.catalog]
    if args.model in JOINT:
        source_paths.append(Path(__file__).with_name('chexchonet_joint_queries.py'))
    sources = {str(p.resolve()): sha(p) for p in source_paths}
    for task in inputs:
        sources.update({task['entry']['manifest']: task['entry']['sha256'],
                        task['entry']['support_manifest']: task['entry']['support_manifest_sha256']})
    write(args.output / 'source_sha256.json', sources)
    write(args.output / 'status.json', dict(status='loading', model=args.model, started_at=now(), smoke=args.smoke))
    native = load_adapter(args.model_root, 'cuda')
    groups = [inputs] if args.model in JOINT else [[x] for x in inputs]
    outcomes = {}
    started = time.time()
    for group in groups:
        names = [x['name'] for x in group]
        classes = [x['classes'][0] for x in group]
        try:
            for item in group:
                out = args.output / item['name']
                out.mkdir(exist_ok=False)
                write(out / 'manifest.json', item['supports'])
            if hasattr(native, 'set_dataset'):
                native.set_dataset(group[0]['data'])
            if hasattr(native, 'check_classes'):
                native.check_classes(classes)
            write(args.output / 'status.json', dict(status='fitting_support_queries', model=args.model, datasets=names))
            with torch.inference_mode():
                if args.model in JOINT:
                    from chexchonet_joint_queries import build_joint_query, check_joint
                    query, individual, fits = build_joint_query(args.model, native, group, args.batch_size)
                else:
                    query = load_query(args.model, native)
                    support = group[0]['support_rows']
                    fits = [query.fit([r['image'] for r in support], [r['labels'] for r in support],
                                       classes, batch_size=args.batch_size, task='classification')]
            for task, fit in zip(group, fits):
                write(args.output / task['name'] / 'query_adapter.json', fit)
            arrays = {branch: [] for branch in BRANCHES}
            rows = group[0]['rows'][:args.smoke] if args.smoke else group[0]['rows']
            begin = time.time()
            last_update = 0
            checks = None
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                paths = [r['image'] for r in batch]
                with torch.inference_mode():
                    identity = paired(query, 'predict', paths, classes, 'identity')
                    center = paired(query, 'predict', paths, classes, 'center80')
                    if offset == 0:
                        checks = {'native': check_native(identity['baseline'], native.predict(paths, classes))}
                        if args.model in JOINT:
                            checks['joint_vs_individual'] = check_joint(query, individual, paths, classes)
                        write(args.output / ('_'.join(names) + '_parity.json'), checks)
                values = {'original': identity['baseline'], 'query_refined': identity['refined'],
                          'tta_center_original': (1.0-view_weight) * np.asarray(identity['baseline']) + view_weight * np.asarray(center['baseline']),
                          'tta_center_refined': (1.0-view_weight) * np.asarray(identity['refined']) + view_weight * np.asarray(center['refined'])}
                for branch, value in values.items():
                    value = np.asarray(value)
                    if value.shape != (len(batch), len(classes)) or not np.isfinite(value).all():
                        raise ValueError('Invalid native prediction values')
                    arrays[branch].append(value)
                if time.time() - last_update > 15 or offset + len(batch) == len(rows):
                    progress = dict(status='running', model=args.model, datasets=names,
                        done=offset + len(batch), total=len(rows), elapsed_seconds=time.time() - begin, updated_at=now())
                    write(args.output / 'status.json', progress)
                    print(json.dumps(progress), flush=True)
                    last_update = time.time()
            joined = {key: np.concatenate(value) for key, value in arrays.items()}
            for j, task in enumerate(group):
                out = args.output / task['name']
                selected = task['rows'][:len(rows)]
                labels = np.asarray([r['labels'] for r in selected], dtype=np.uint8)
                saved = dict(labels=labels, image_paths=np.asarray([r['image'] for r in selected]),
                             patient_ids=np.asarray([r['patient_id'] for r in selected]), class_names=np.asarray(task['classes']))
                metrics = {}
                for branch, values in joined.items():
                    saved[branch] = values[:, j:j+1]
                    metrics[branch] = classification_metrics(labels, saved[branch], task['classes'])
                np.savez_compressed(out / 'predictions.npz', **saved)
                base, ours = metrics['original'], metrics['tta_center_refined']
                delta = {key: None if base[key] is None else 100 * (ours[key] - base[key])
                         for key in ('macro_auroc', 'macro_auprc', 'macro_average_precision')}
                result = dict(status='complete', model=args.model, dataset=task['name'], task='classification',
                    rows=len(rows), smoke=bool(args.smoke), class_names=task['classes'], support_count=10,
                    eta=float(query.metadata.get('eta', os.environ.get('QUERY_REFINEMENT_ETA', '.3'))),
                    crop_ratio=float(os.environ.get('QUERY_REFINEMENT_CROP_RATIO', '.8')), view_weight=view_weight,
                    branches=metrics, delta_pp=delta, native_parity=checks,
                    metadata=query.metadata, original_adapter_metadata=native.metadata,
                    elapsed_seconds=time.time() - begin, source_sha256=sources,
                    test_labels_used_for_query_fit=False, no_gradient_training=True,
                    protocol='Exact task-specific 5-positive/5-negative support patients; sensitivity-grid eta/crop/view-weight settings recorded; native query norm/negative prompts/scoring retained.',
                    prompt_provenance='Fixed disease semantics. MedKLIP/MAVL use an explicitly reconstructed query-resource extension, not author-released CheXchoNet resources.')
                write(out / 'result.json', result)
                outcomes[task['name']] = 'complete'
                print('COMPLETE', args.model, task['name'], json.dumps(delta), flush=True)
            del query, fits, arrays, joined
            torch.cuda.empty_cache()
        except Exception as exc:
            traceback.print_exc()
            for task in group:
                out = args.output / task['name']
                out.mkdir(exist_ok=True)
                write(out / 'result.json', dict(status='failed', model=args.model, dataset=task['name'],
                    error=repr(exc), traceback=traceback.format_exc(), rows=7407, smoke=bool(args.smoke)))
                outcomes[task['name']] = 'failed'
    changed = [p for p, digest in sources.items() if sha(p) != digest]
    status = 'complete' if set(outcomes.values()) == {'complete'} and not changed else 'failed'
    write(args.output / 'status.json', dict(status=status, model=args.model, outcomes=outcomes,
        changed_sources=changed, elapsed_seconds=time.time()-started, smoke=bool(args.smoke), finished_at=now()))
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.smoke < 0:
        parser.error('Invalid batch or smoke size')
    sys.exit(main(args))
