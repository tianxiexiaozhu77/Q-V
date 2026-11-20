#!/usr/bin/env python3
"""Eleven comparators on the repaired, fixed five-support protocol (GPUs 4--7)."""
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

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
sys.path.insert(0, str(MEDICAL / 'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu

ENVS = {
    'BiomedCLIP': MEDICAL / 'baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/python_local.sh',
    'AlphaRAD': MEDICAL / 'AlphaRAD/local_setup/python_local.sh',
}


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def base_env():
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1'}


def signal_group(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol_v1/catalog.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[4, 5, 6, 7])
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {4, 5, 6, 7}:
        parser.error('Only distinct physical GPU IDs 4--7 are authorized')
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()
    catalog = json.loads(args.protocol.read_text())
    if catalog['status'] != 'frozen' or catalog['support_seeds'] != [42, 43, 44, 45, 46]:
        raise ValueError('Expected frozen repaired five-support protocol')
    for name in ('cache_multiseed.py', 'evaluate_multiseed.py', 'summarize_multiseed.py'):
        if not (HERE / name).is_file():
            raise FileNotFoundError(HERE / name)
    sources = {str(p.resolve()): sha(p) for p in HERE.glob('*.py')}
    sources[str(args.protocol)] = sha(args.protocol)
    extra_sources = [HERE / 'cache_reuse_sources_v1.json',
                     MEDICAL / 'experiments/fewshot_native_query/cache_native_common.py',
                     MEDICAL / 'experiments/fewshot_native_query/cache_alpharad.py',
                     MEDICAL / 'experiments/chexchonet_stability_20261024/cluster_bootstrap.py',
                     MEDICAL / 'baselines/local_setup/evaluate_baseline_buffered.py',
                     MEDICAL / 'AlphaRAD/local_setup/run_pretrained.py']
    for source in extra_sources:
        sources[str(source)] = sha(source)
    for spec in catalog['algorithm_sources'].values():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Original algorithm changed: ' + spec['path'])
        sources[spec['path']] = spec['sha256']
    for spec in catalog['datasets'].values():
        for item in [spec, *spec['seeds'].values()]:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Protocol changed: ' + item['path'])
            sources[item['path']] = item['sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    state = dict(status='running', started_at=now(), controller_pid=os.getpid(),
                 gpus=args.gpus, protocol=str(args.protocol), protocol_sha256=sha(args.protocol),
                 sources=sources, smoke_only=args.smoke_only,
                 jobs={m+'_'+d: dict(model=m, dataset=d, stage='smoke', status='pending')
                       for d in ('rsna_pneumonia', 'chestxdet10') for m in ENVS})
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        state['stop_signal'] = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        state['updated_at'] = now()
        write(args.output / 'status.json', state)

    def start(key, stage, command, env, lease=None, gpu=None, gpu_check=None):
        log = (args.output / 'logs' / f'{key}_{stage}.log').open('w')
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=MEDICAL, env=env, start_new_session=True)
        except BaseException:
            log.close()
            if lease is not None:
                release_gpu(lease)
            raise
        active[key] = dict(proc=proc, log=log, lease=lease, gpu=gpu, stage=stage)
        state['jobs'][key].update(status='running', stage=stage, pid=proc.pid,
                                  gpu=gpu, command=command, stage_started_at=now())
        if gpu_check is not None:
            state['jobs'][key]['last_gpu_check'] = gpu_check
        print('Started', key, stage, 'GPU' if gpu is not None else 'CPU', gpu, 'PID', proc.pid, flush=True)

    try:
        save()
        while active or any(j['status'] == 'pending' for j in state['jobs'].values()):
            if stopping:
                raise InterruptedError('Controller stopping; terminating only its owned process groups')
            for key, item in list(active.items()):
                code = item['proc'].poll()
                if code is None:
                    continue
                item['log'].close()
                if item['lease'] is not None:
                    release_gpu(item['lease'])
                del active[key]
                stage = item['stage']
                job = state['jobs'][key]
                if stage == 'evaluate':
                    receipt = args.output / key / 'evaluation/status.json'
                else:
                    receipt = args.output / (key + ('_smoke' if stage == 'smoke' else '')) / 'cache/status.json'
                expected = 'smoke_complete' if stage == 'smoke' else 'complete'
                ok = (code == 0 and receipt.exists()
                      and json.loads(receipt.read_text())['status'] == expected)
                job.setdefault('stages', {})[stage] = dict(returncode=code, success=ok,
                    finished_at=now(), status_file=str(receipt))
                if ok and stage == 'smoke' and not args.smoke_only:
                    job.update(status='pending', stage='full')
                elif ok and stage == 'full':
                    job.update(status='pending', stage='evaluate')
                else:
                    job.update(status='complete' if ok else 'failed')
                print(key, stage, job['status'], flush=True)
            for key, job in state['jobs'].items():
                if job['status'] == 'pending' and job['stage'] == 'evaluate':
                    command = ['bash', str(ENVS[job['model']]), str(HERE / 'evaluate_multiseed.py'),
                               '--protocol', str(args.protocol), '--model', job['model'],
                               '--dataset', job['dataset'], '--cache', str(args.output / key / 'cache'),
                               '--output', str(args.output / key / 'evaluation')]
                    start(key, 'evaluate', command, base_env())
            for gpu in args.gpus:
                if any(item['gpu'] == gpu for item in active.values()):
                    continue
                key = next((k for k, j in state['jobs'].items()
                            if j['status'] == 'pending' and j['stage'] in ('smoke', 'full')), None)
                if key is None:
                    break
                lease, info = reserve_gpu(gpu)
                if lease is None:
                    state.setdefault('last_gpu_checks', {})[str(gpu)] = info
                    continue
                job = state['jobs'][key]
                stage = job['stage']
                output = args.output / (key + ('_smoke' if stage == 'smoke' else '')) / 'cache'
                command = ['bash', str(ENVS[job['model']]), str(HERE / 'cache_multiseed.py'),
                           '--protocol', str(args.protocol), '--model', job['model'],
                           '--dataset', job['dataset'], '--output', str(output), '--gpu', str(gpu),
                           '--batch-size', '8' if job['model'] == 'AlphaRAD' else '16',
                           '--smoke', '16' if stage == 'smoke' else '0']
                env = {**base_env(), 'CUDA_VISIBLE_DEVICES': info['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
                start(key, stage, command, env, lease, gpu, info)
            save()
            if active or any(j['status'] == 'pending' for j in state['jobs'].values()):
                time.sleep(2)
        ok = all(j['status'] == 'complete' for j in state['jobs'].values())
        if ok and not args.smoke_only:
            state['status'] = 'paired_bootstrap'
            command = [sys.executable, str(HERE / 'summarize_multiseed.py'), '--run', str(args.output),
                       '--protocol', str(args.protocol), '--replicates', str(catalog['bootstrap_replicates']),
                       '--seed', str(catalog['bootstrap_seed'])]
            state['jobs']['summary'] = dict(status='pending', stage='summary')
            start('summary', 'summary', command, base_env())
            save()
            while active['summary']['proc'].poll() is None:
                if stopping:
                    raise InterruptedError('Stopping summary')
                time.sleep(2)
            code = active['summary']['proc'].returncode
            active['summary']['log'].close()
            del active['summary']
            result_path = args.output / 'RESULTS.json'
            ok = code == 0 and result_path.is_file() and json.loads(result_path.read_text())['status'] == 'complete'
            state['jobs']['summary'].update(status='complete' if ok else 'failed', returncode=code, finished_at=now())
        changed = [path for path, digest in sources.items() if not Path(path).is_file() or sha(path) != digest]
        state.update(status='complete' if ok and not changed else 'failed', finished_at=now(), changed_sources=changed)
        save()
        print('Campaign', state['status'], str(args.output), flush=True)
        return 0 if state['status'] == 'complete' else 1
    except BaseException as exc:
        for item in active.values():
            signal_group(item['proc'], signal.SIGTERM)
        deadline = time.monotonic() + 20
        while any(item['proc'].poll() is None for item in active.values()) and time.monotonic() < deadline:
            time.sleep(.2)
        for key, item in active.items():
            if item['proc'].poll() is None:
                signal_group(item['proc'], signal.SIGKILL)
            item['proc'].wait()
            item['log'].close()
            if item['lease'] is not None:
                release_gpu(item['lease'])
            state['jobs'][key].update(status='interrupted', returncode=item['proc'].returncode)
        state.update(status='interrupted' if stopping else 'failed', error=repr(exc), finished_at=now())
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
