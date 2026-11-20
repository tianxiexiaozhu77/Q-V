#!/usr/bin/env python3
"""Run pre-fixed Q-F native-aggregation interventions on CheXchoNet.

CARZero/KAD/MedKLIP/MAVL have a fixed classifier, not an independent
query-conditioned final readout. Their control is fixed_aggregation (F).
All five inherited supports and three tasks are retained. One visual forward
is shared across all task/seed predictions within a test-image batch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('inherited_chexcho_stability',
    ROOT / 'experiments/chexchonet_stability_20261024/infer_multiseed.py')
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)
from baseline_intervention import SUPPORTED, SOURCES, DECODER_SOURCES, create_intervention, _shared_visual_encoding

SEEDS, TASKS = stability.SEEDS, stability.TASKS
BRANCHES = ('original', 'fixed_aggregation', 'query')
sha, write = stability.sha, stability.write


def progress(args, stage, started, done=0, total=0):
    item = dict(status=stage, model=args.model, contrast='Q-F', done=done, total=total,
                elapsed_seconds=time.time() - started, smoke=args.smoke)
    write(args.output / 'status.json', item)
    print(json.dumps(item), flush=True)


def _historical_original(model, rows, original):
    checks = {}
    for j, task in enumerate(TASKS):
        path = ROOT / 'baselines/runs/chexchonet_20261024_000447' / (model + '_full') / ('chexchonet_' + task) / 'predictions.npz'
        with np.load(path, allow_pickle=False) as old:
            count = len(rows)
            expected_paths = [str(Path(p).resolve()) for p in old['image_paths'][:count]]
            if expected_paths != [r['image_path'] for r in rows]:
                raise ValueError('Historical Original image order differs')
            if list(old['patient_ids'][:count].astype(str)) != [r['patient_id'] for r in rows]:
                raise ValueError('Historical Original patient order differs')
            expected = old['original'][:count, 0]
            actual = original[:, j]
            # Different historical batching can produce small floating-point
            # differences; current same-batch native parity uses stricter bounds.
            error = float(np.max(np.abs(actual - expected)))
            if not np.allclose(actual, expected, rtol=1e-4, atol=2e-5):
                raise ValueError(f'{model}/{task} historical Original mismatch: {error}')
            checks[task] = dict(passed=True, source=str(path), source_sha256=sha(path),
                                max_abs_error=error, rtol=1e-4, atol=2e-5)
    return checks


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = stability.load_protocol(args.protocol)
    experiment = protocol['master'].get('experiment', {})
    if experiment.get('views') != ['identity']:
        raise ValueError('Require this campaign\'s fixed single-view protocol')
    rows = protocol['test'][:args.smoke] if args.smoke else protocol['test']
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Assigned CUDA GPU unavailable')
    sys.path.insert(0, str(ROOT / 'baselines/local_setup'))
    from evaluate_baseline_buffered import load_adapter, classification_metrics
    from evaluate_query_transfer import load_query
    model_root = ROOT / 'baselines' / args.model
    if args.model in ('MedKLIP', 'MAVL'):
        model_root = ROOT / 'baselines/protocols/chexchonet' / args.model
    source_paths = [Path(__file__), HERE / 'baseline_intervention.py', Path(stability.__file__),
                    ROOT / SOURCES[args.model], ROOT / DECODER_SOURCES[args.model],
                    ROOT / 'baselines' / args.model / 'local_setup/query_refinement_adapter.py',
                    ROOT / 'baselines' / args.model / 'local_setup/adapter.py',
                    model_root / 'local_setup/adapter.py',
                    ROOT / 'baselines/local_setup/query_refinement_common.py']
    if args.model in ('MedKLIP', 'MAVL'):
        source_paths.append(ROOT / 'baselines' / args.model / 'local_setup/chexchonet_adapter.py')
    sources = {**protocol['sources'], **{str(p.resolve()): sha(p) for p in source_paths}}
    write(args.output / 'source_sha256.json', sources)
    write(args.output / 'input_audit.json', dict(
        protocol=str(protocol['path']), protocol_sha256=sha(protocol['path']),
        fixed_test_images=7407, fixed_test_patients=2470, inference_images=len(rows),
        class_names=protocol['classes'], task_keys=list(TASKS), support_seeds=list(SEEDS),
        support_test_patient_overlap=0, support_images_per_task=10,
        supports_inherited_without_resampling=True, eta=.3, views=['identity'],
        control_name='fixed_aggregation', comparison='Q-F', not_readout_only_R=True))
    progress(args, 'loading_model', started)
    native = load_adapter(model_root, 'cuda')
    if hasattr(native, 'set_dataset'):
        native.set_dataset({'dataset': 'chexchonet_multitask', 'rows': []})
    adapters, fit_metadata = {}, {}
    progress(args, 'fitting_all_support_queries', started)
    for seed in SEEDS:
        folder = args.output / f'seed_{seed}'
        folder.mkdir()
        write(folder / 'support_manifest.json', dict(seed=seed,
            task_support_rows=protocol['supports'][seed], eta=.3,
            test_labels_used_for_fit=False, no_test_selection=True))
        adapters[seed] = {}
        fit_metadata[str(seed)] = {}
        for j, task in enumerate(TASKS):
            names = [protocol['classes'][j]]
            query = load_query(args.model, native)
            intervention = create_intervention(args.model, native, query)
            support = protocol['supports'][seed][task]
            fit = intervention.fit([r['image_path'] for r in support], [[r[task]] for r in support],
                                   names, batch_size=args.batch_size)
            write(folder / f'query_fit_{task}.json', fit)
            adapters[seed][task] = intervention
            fit_metadata[str(seed)][task] = intervention.metadata
    arrays = {seed: {branch: np.empty((len(rows), len(TASKS)), dtype=np.float32)
                     for branch in BRANCHES} for seed in SEEDS}
    audits = {}
    last_update = 0.0
    progress(args, 'inference', started, 0, len(rows))
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        paths = [r['image_path'] for r in batch]
        with torch.inference_mode(), _shared_visual_encoding(native, args.model):
            for seed in SEEDS:
                for j, task in enumerate(TASKS):
                    intervention = adapters[seed][task]
                    out = intervention.predict_all(paths, [protocol['classes'][j]], verify=offset == 0)
                    for branch, value in out.items():
                        if value.shape != (len(batch), 1) or not np.isfinite(value).all():
                            raise ValueError('Malformed or nonfinite intervention predictions')
                        arrays[seed][branch][offset:offset + len(batch), j] = value[:, 0]
                    if offset == 0:
                        audits[f'{seed}/{task}'] = intervention.last_audit
        if offset == 0:
            write(args.output / 'native_parity.json', audits)
        if time.time() - last_update > 15 or offset + len(batch) == len(rows):
            progress(args, 'inference', started, offset + len(batch), len(rows))
            last_update = time.time()
    for seed in SEEDS[1:]:
        np.testing.assert_array_equal(arrays[seed]['original'], arrays[SEEDS[0]]['original'])
    audits['Original_bitwise_equal_across_seeds'] = True
    audits['historical_Original'] = _historical_original(args.model, rows, arrays[SEEDS[0]]['original'])
    labels = np.asarray([[r[task] for task in TASKS] for r in rows], dtype=np.uint8)
    # Save every seed's predictions before computing any test metrics.
    for seed in SEEDS:
        np.savez_compressed(args.output / f'seed_{seed}' / 'predictions.npz',
            labels=labels, image_paths=np.asarray([r['image_path'] for r in rows]),
            patient_ids=np.asarray([r['patient_id'] for r in rows]),
            class_names=np.asarray(protocol['classes']), task_keys=np.asarray(TASKS),
            seed=np.asarray(seed), **arrays[seed])
    seed_results = {}
    for seed in SEEDS:
        seed_results[str(seed)] = dict(status='smoke_complete' if args.smoke else 'complete',
            model=args.model, seed=seed, rows=len(rows), contrast='Q-F',
            branches={branch: classification_metrics(labels, values, protocol['classes'])
                      for branch, values in arrays[seed].items()},
            prediction_sha256=sha(args.output / f'seed_{seed}' / 'predictions.npz'))
        write(args.output / f'seed_{seed}' / 'result.json', seed_results[str(seed)])
    changed = [path for path, digest in sources.items() if sha(path) != digest]
    if changed:
        raise RuntimeError('Protocol or source changed during execution: ' + repr(changed))
    write(args.output / 'native_parity.json', audits)
    write(args.output / 'metadata.json', dict(native=native.metadata, interventions=fit_metadata))
    status = 'smoke_complete' if args.smoke else 'complete'
    write(args.output / 'result.json', dict(status=status, model=args.model, rows=len(rows),
        patients=len({r['patient_id'] for r in rows}), support_seeds=list(SEEDS),
        class_names=protocol['classes'], task_keys=list(TASKS), control_name='fixed_aggregation',
        comparison='Q-F', readout_only_R_supported=False, views=['identity'], eta=.3,
        seed_results=seed_results, native_parity=audits, no_test_selection=True,
        batch_size=args.batch_size, elapsed_seconds=time.time() - started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__))
    progress(args, status, started, len(rows), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=SUPPORTED)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2,
                        help='Accepted for campaign CLI compatibility; native adapters load images synchronously.')
    parser.add_argument('--smoke', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error('Invalid batch size, workers or smoke count')
    args.output = args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir():
            write(args.output / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
