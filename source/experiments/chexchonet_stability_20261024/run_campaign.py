#!/usr/bin/env python3
"""Run the predeclared two-model CheXchoNet stability study on idle GPUs 4--7."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
sys.path.insert(0, str(MEDICAL / 'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu

ENVS = {
    'BiomedCLIP': MEDICAL / 'baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/python_local.sh',
    'AlphaRAD': MEDICAL / 'AlphaRAD/local_setup/python_local.sh',
}


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now().isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT / 'protocol_v1/catalog.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=[4, 5, 6, 7])
    p.add_argument('--smoke-only', action='store_true')
    a = p.parse_args()
    if not a.gpus or not set(a.gpus) <= set(range(4, 8)) or len(set(a.gpus)) != len(a.gpus):
        p.error('Use distinct authorized physical GPUs 4--7')
    a.protocol = a.protocol.resolve()
    a.output = a.output.resolve()
    if not a.protocol.is_file():
        raise FileNotFoundError(a.protocol)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'logs').mkdir()
    sources = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
    state = dict(status='running', started_at=now(), pid=os.getpid(), gpus=a.gpus,
                 protocol=str(a.protocol), protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(),
                 sources=sources, smoke_only=a.smoke_only,
                 jobs={m: dict(status='pending', stage='smoke') for m in ENVS})
    write(a.output / 'status.json', state)
    active = {}
    try:
        while active or any(j['status'] == 'pending' for j in state['jobs'].values()):
            for gpu, (proc, lease, log, model, stage) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                release_gpu(lease)
                del active[gpu]
                job = state['jobs'][model]
                result = a.output / (model + ('_smoke' if stage == 'smoke' else '')) / 'status.json'
                expected_status = 'smoke_complete' if stage == 'smoke' else 'complete'
                ok = code == 0 and result.exists() and json.loads(result.read_text())['status'] == expected_status
                job.setdefault('stages', {})[stage] = dict(returncode=code, success=ok, finished_at=now(), status_file=str(result))
                if ok and stage == 'smoke' and not a.smoke_only:
                    job.update(status='pending', stage='full')
                else:
                    job.update(status='complete' if ok else 'failed')
                print(model, stage, job['status'], flush=True)
            for gpu in a.gpus:
                if gpu in active:
                    continue
                model = next((m for m, j in state['jobs'].items() if j['status'] == 'pending'), None)
                if model is None:
                    break
                lease, info = reserve_gpu(gpu)
                if lease is None:
                    state.setdefault('last_gpu_checks', {})[str(gpu)] = info
                    continue
                job = state['jobs'][model]
                stage = job['stage']
                output = a.output / (model + ('_smoke' if stage == 'smoke' else ''))
                command = ['bash', str(ENVS[model]), str(ROOT / 'infer_multiseed.py'),
                           '--model', model, '--protocol', str(a.protocol), '--output', str(output),
                           '--batch-size', '8',
                           '--smoke', '16' if stage == 'smoke' else '0']
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': info['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                       'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
                log = (a.output / 'logs' / f'{model}_{stage}.log').open('w')
                try:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                            cwd=MEDICAL, start_new_session=True)
                except BaseException:
                    log.close()
                    release_gpu(lease)
                    raise
                job.update(status='running', pid=proc.pid, gpu=gpu, gpu_check=info, command=command, started_at=now())
                active[gpu] = (proc, lease, log, model, stage)
                print('Started', model, stage, 'GPU', gpu, 'PID', proc.pid, flush=True)
            write(a.output / 'status.json', state)
            if active or any(j['status'] == 'pending' for j in state['jobs'].values()):
                time.sleep(5)
        ok = all(j['status'] == 'complete' for j in state['jobs'].values())
        if ok and not a.smoke_only:
            state['status'] = 'patient_bootstrap'
            write(a.output / 'status.json', state)
            command = [sys.executable, str(ROOT / 'summarize_stability.py'), '--run', str(a.output),
                       '--protocol', str(a.protocol)]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            with (a.output / 'logs/bootstrap.log').open('w') as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, cwd=MEDICAL, env=env)
            state['bootstrap_returncode'] = code
            ok = code == 0
        changed = [str(f) for f, digest in sources.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest() != digest]
        state.update(status='complete' if ok and not changed else 'failed', finished_at=now(), changed_sources=changed)
        write(a.output / 'status.json', state)
        return 0 if state['status'] == 'complete' else 1
    except BaseException:
        state.update(status='controller_interrupted_workers_not_terminated', updated_at=now())
        write(a.output / 'status.json', state)
        raise


if __name__ == '__main__':
    sys.exit(main())
