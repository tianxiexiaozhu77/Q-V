#!/usr/bin/env python3
"""Run five-support-seed QaTa inference, validation selection, and paired scoring.

Use --preflight for read-only CPU checks. GPU inference acquires an idle physical
GPU 4--7 by UUID. Every validation threshold is frozen before test-mask scoring.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
PYTHON = MEDICAL / 'AlphaRAD/local_setup/python_local.sh'
SEEDS = [42, 43, 44, 45, 46]
BRANCHES = ['original', 'positive_query', 'query_refined',
            'tta_center_original', 'tta_center_refined']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def assert_frozen(snapshot):
    changed = [p for p, expected in snapshot.items()
               if not Path(p).is_file() or sha(p) != expected]
    if changed:
        raise ValueError('Frozen files changed during campaign: ' + repr(changed))


def validate_frozen_thresholds(path):
    """Fail closed before test scoring unless all 25 choices were serialized."""
    data = json.loads(Path(path).read_text())
    if data.get('status') != 'frozen_before_fixed_test_scoring':
        raise ValueError('Threshold selection did not finish')
    if data.get('seeds') != SEEDS or data.get('branch_names') != BRANCHES:
        raise ValueError('Threshold seeds or branches differ from the fixed design')
    choices = data.get('thresholds', {})
    if set(choices) != set(map(str, SEEDS)):
        raise ValueError('Expected thresholds for all five support seeds')
    grid = data.get('threshold_grid', [])
    if len(grid) != 101 or any(not math.isfinite(float(v)) or abs(float(v) - i / 100) > 1e-12
                               for i, v in enumerate(grid)):
        raise ValueError('Unexpected validation threshold grid')
    for seed in SEEDS:
        if set(choices[str(seed)]) != set(BRANCHES):
            raise ValueError(f'Expected five threshold choices for seed {seed}')
        for branch in BRANCHES:
            choice = choices[str(seed)][branch]
            index = choice.get('threshold_index')
            value = float(choice.get('threshold', float('nan')))
            if type(index) is not int or index not in range(len(grid)):
                raise ValueError(f'Invalid threshold index: seed {seed}, {branch}')
            if not math.isfinite(value) or abs(value - grid[index]) > 1e-12:
                raise ValueError(f'Invalid threshold value: seed {seed}, {branch}')
    return data


def terminate_owned_worker(process):
    """Keep our lease until our own child process group has stopped."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol_v1/catalog.json')
    parser.add_argument('--output', type=Path,
                        help='New directory only; defaults to runs/run_<timestamp>')
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5, 6, 7])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true',
                      help='Read-only CPU asset/protocol/source checks; do not reserve a GPU')
    mode.add_argument('--smoke-only', action='store_true',
                      help='GPU checks on 16 images per split; no thresholds or test metrics')
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {4, 5, 6, 7}:
        parser.error('Use distinct authorized physical GPUs 4--7')
    args.protocol = args.protocol.resolve()
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    # Parsing does not import model code or emit bytecode files.
    for script in sorted(ROOT.glob('*.py')):
        ast.parse(script.read_text(), filename=str(script))
    base_env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    # Always validate before creating results or reserving a device. The worker's
    # preflight verifies identities and existing source files without loading masks.
    command = ['bash', str(PYTHON), str(ROOT / 'infer_maps.py'),
               '--protocol', str(args.protocol), '--preflight']
    subprocess.run(command, cwd=MEDICAL,
                   env={**base_env, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
    if args.preflight:
        return 0

    # source_snapshot covers the imported native inference/metric dependencies.
    from common import source_snapshot
    snapshot = dict(source_snapshot())
    for path in [args.protocol, ROOT / 'run.sh', PYTHON,
                 MEDICAL / 'AlphaRAD/local_setup/activate.sh',
                 MEDICAL / 'AlphaRAD/local_setup/run_pretrained.py']:
        snapshot[str(path)] = sha(path)
    for pattern in ('*.json', '*.csv'):
        for path in sorted(args.protocol.parent.rglob(pattern)):
            snapshot[str(path)] = sha(path)

    args.output = (args.output or ROOT / 'runs' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    state = dict(status='preparing', started_at=now(), pid=os.getpid(),
                 authorized_gpus=args.gpus, protocol=str(args.protocol),
                 protocol_sha256=sha(args.protocol), frozen_inputs=snapshot,
                 stages={}, seeds=SEEDS, branch_names=BRANCHES,
                 selection_rule='Freeze all 25 validation-only thresholds before opening test masks.',
                 no_gradient_training=True, smoke_only=args.smoke_only)
    write(args.output / 'status.json', state)
    # Importing the lease helper alone does not query or reserve a GPU.
    sys.path.insert(0, str(MEDICAL / 'AlphaRAD/local_setup'))
    from run_pretrained import reserve_gpu, release_gpu
    lease = None
    active_process = None

    def phase(name, script, arguments, visible=''):
        nonlocal active_process
        assert_frozen(snapshot)
        command = ['bash', str(PYTHON), str(ROOT / script), *arguments]
        env = {**base_env, 'CUDA_VISIBLE_DEVICES': visible}
        record = dict(command=command, started_at=now(), status='running')
        state['stages'][name] = record
        state['status'] = name
        with (args.output / 'logs' / (name + '.log')).open('x') as handle:
            active_process = subprocess.Popen(command, cwd=MEDICAL, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = active_process.pid
            write(args.output / 'status.json', state)
            print('START', name, 'PID', active_process.pid, flush=True)
            while True:
                try:
                    code = active_process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    assert_frozen(snapshot)
            active_process = None
        record.update(returncode=code, finished_at=now(),
                      status='complete' if code == 0 else 'failed')
        write(args.output / 'status.json', state)
        if code:
            raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')
        assert_frozen(snapshot)
        print('COMPLETE', name, flush=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while lease is None:
            assert_frozen(snapshot)
            for gpu in args.gpus:
                lease, info = reserve_gpu(gpu)
                state.setdefault('gpu_checks', {})[str(gpu)] = info
                if lease is not None:
                    state['gpu'] = info
                    print('RESERVED physical GPU', gpu, info['uuid'], flush=True)
                    break
            if lease is None:
                state['status'] = 'waiting_for_idle_gpu'
                write(args.output / 'status.json', state)
                time.sleep(10)
        visible = state['gpu']['uuid']
        common_args = ['--protocol', str(args.protocol), '--batch-size', '8']
        phase('smoke', 'infer_maps.py', [*common_args, '--output', str(args.output / 'smoke'),
                                       '--smoke', '16'], visible)
        smoke = json.loads((args.output / 'smoke/status.json').read_text())
        if smoke.get('status') != 'smoke_complete':
            raise ValueError('Worker did not finish the required smoke check')
        if args.smoke_only:
            state.update(status='smoke_complete', finished_at=now())
            write(args.output / 'status.json', state)
            return 0
        maps = args.output / 'maps'
        phase('full_inference', 'infer_maps.py',
              [*common_args, '--output', str(maps), '--smoke', '0'], visible)
        full = json.loads((maps / 'status.json').read_text())
        if full.get('status') != 'complete':
            raise ValueError('Worker did not finish the full validation/test image inference')
        release_gpu(lease)
        lease = None
        state['gpu_released_at'] = now()
        write(args.output / 'status.json', state)
        thresholds = args.output / 'thresholds.json'
        phase('select_validation_thresholds', 'select_thresholds.py',
              ['--protocol', str(args.protocol), '--maps', str(maps),
               '--output', str(thresholds)])
        validate_frozen_thresholds(thresholds)
        state['frozen_thresholds'] = dict(path=str(thresholds), sha256=sha(thresholds),
                                          frozen_at=now(), choices=25)
        snapshot[str(thresholds)] = state['frozen_thresholds']['sha256']
        write(args.output / 'status.json', state)
        phase('evaluate_fixed_test', 'evaluate_fixed_test.py',
              ['--protocol', str(args.protocol), '--maps', str(maps),
               '--thresholds', str(thresholds), '--output', str(args.output / 'evaluation')])
        assert_frozen(snapshot)
        state.update(status='complete', finished_at=now(), changed_sources=[])
        write(args.output / 'status.json', state)
        print('COMPLETE campaign:', args.output, flush=True)
        return 0
    except BaseException as error:
        state.update(status='failed_or_interrupted', updated_at=now(), error=repr(error),
                     traceback=traceback.format_exc())
        write(args.output / 'status.json', state)
        raise
    finally:
        # Only the child group created above may be terminated; other GPU jobs
        # are never touched, including jobs observed while waiting for a lease.
        terminate_owned_worker(active_process)
        if lease is not None:
            release_gpu(lease)
            state['gpu_released_at'] = now()
            write(args.output / 'status.json', state)


if __name__ == '__main__':
    sys.exit(main())
