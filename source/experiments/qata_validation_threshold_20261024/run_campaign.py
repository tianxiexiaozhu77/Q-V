#!/usr/bin/env python3
"""QaTa: infer validation maps, freeze thresholds, then score the fixed test.

Reuse frozen phase6 queries and test maps. Never select a threshold using test
labels. GPU inference is limited to an idle authorized physical GPU 4--7.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
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
sys.path.insert(0, str(MEDICAL / 'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def now():
    return datetime.now().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol_v1/catalog.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5, 6, 7])
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {4, 5, 6, 7}:
        parser.error('Use distinct authorized physical GPUs 4--7')
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    state = dict(status='preparing', started_at=now(), pid=os.getpid(), gpus=args.gpus,
                 protocol=str(args.protocol), protocol_sha256=sha(args.protocol),
                 source_sha256={str(p): sha(p) for p in ROOT.glob('*.py')}, stages={},
                 selection_rule='Choose each branch threshold on validation only, serialize it before opening test masks.',
                 reused_test_maps=True, no_gradient_training=True)
    write(args.output / 'status.json', state)
    lease = None
    active_process = None
    base_env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
                'TRANSFORMERS_OFFLINE': '1', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}

    def phase(name, script, arguments, visible=''):
        nonlocal active_process
        command = ['bash', str(PYTHON), str(ROOT / script), *arguments]
        env = {**base_env, 'CUDA_VISIBLE_DEVICES': visible}
        record = dict(command=command, started_at=now(), status='running')
        state['stages'][name] = record
        state['status'] = name
        with (args.output / 'logs' / (name + '.log')).open('x') as handle:
            process = subprocess.Popen(command, cwd=MEDICAL, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            active_process = process
            record['pid'] = process.pid
            write(args.output / 'status.json', state)
            print('START', name, 'PID', process.pid, flush=True)
            code = process.wait()
            active_process = None
        record.update(returncode=code, finished_at=now(), status='complete' if code == 0 else 'failed')
        write(args.output / 'status.json', state)
        if code:
            raise RuntimeError(f'{name} failed ({code}); see logs/{name}.log')
        print('COMPLETE', name, flush=True)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    try:
        while lease is None:
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
        common = ['--protocol', str(args.protocol), '--batch-size', '8']
        phase('smoke', 'infer_maps.py', [*common, '--output', str(args.output / 'smoke'), '--smoke', '16'], visible)
        smoke = json.loads((args.output / 'smoke/status.json').read_text())
        if smoke['status'] != 'smoke_complete':
            raise ValueError('Worker did not finish the required smoke check')
        if args.smoke_only:
            state.update(status='smoke_complete', finished_at=now())
            write(args.output / 'status.json', state)
            return 0
        phase('validation_inference', 'infer_maps.py',
              [*common, '--output', str(args.output / 'validation_maps'), '--smoke', '0'], visible)
        full = json.loads((args.output / 'validation_maps/status.json').read_text())
        if full['status'] != 'complete':
            raise ValueError('Worker did not complete validation inference')
        release_gpu(lease)
        lease = None
        state['gpu_released_at'] = now()
        write(args.output / 'status.json', state)
        thresholds = args.output / 'thresholds.json'
        phase('select_validation_thresholds', 'select_thresholds.py',
              ['--protocol', str(args.protocol), '--maps', str(args.output / 'validation_maps'),
               '--output', str(thresholds)])
        # Bind the serialized choice before executing any test-label scoring.
        state['frozen_thresholds'] = dict(path=str(thresholds), sha256=sha(thresholds), frozen_at=now())
        write(args.output / 'status.json', state)
        cache = args.output / 'validation_maps/test_cache_manifest.json'
        phase('evaluate_fixed_test', 'evaluate_fixed_test.py',
              ['--protocol', str(args.protocol), '--thresholds', str(thresholds),
               '--maps', str(cache), '--output', str(args.output / 'evaluation')])
        if sha(thresholds) != state['frozen_thresholds']['sha256']:
            raise ValueError('Thresholds changed during test evaluation')
        changed = [path for path, expected in state['source_sha256'].items() if sha(path) != expected]
        if sha(args.protocol) != state['protocol_sha256']:
            changed.append(str(args.protocol))
        if changed:
            raise ValueError('Frozen sources changed during run: ' + repr(changed))
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
        # Do not release our GPU lease while an interrupted owned worker runs.
        if active_process is not None and active_process.poll() is None:
            os.killpg(active_process.pid, signal.SIGTERM)
            try:
                active_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(active_process.pid, signal.SIGKILL)
                active_process.wait()
        if lease is not None:
            release_gpu(lease)


if __name__ == '__main__':
    sys.exit(main())
