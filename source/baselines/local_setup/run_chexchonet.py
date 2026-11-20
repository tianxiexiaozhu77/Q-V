#!/usr/bin/env python3
"""Check each frozen CheXchoNet adapter, then evaluate all eight on GPU 4-7."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu
from evaluate_baseline_buffered import sha, write

MODELS = ['CheXzero', 'CARZero', 'MaCo', 'BioViL-T', 'KAD', 'BiomedCLIP', 'MedKLIP', 'MAVL']
MODEL_ROOTS = {name: ROOT / ('protocols/alpharad_binary/' + name if name in ('BiomedCLIP', 'BioViL-T')
    else 'protocols/chexchonet/' + name if name in ('MedKLIP', 'MAVL') else name) for name in MODELS}


def now():
    return datetime.now(timezone.utc).isoformat()


def ready(model):
    root = MODEL_ROOTS[model]
    path = root / 'local_setup/READY.json'
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get('ready') is not True:
        return None
    for key, file in [('adapter_sha256', 'adapter.py'), ('manifest_sha256', 'manifest.json')]:
        if record[key] != sha(root / 'local_setup' / file):
            raise ValueError('Readiness hash mismatch: ' + model + '/' + file)
    return {'path': str(path), 'sha256': sha(path), 'value': record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'evaluation/chexchonet_20261023/catalog.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5, 6, 7])
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or set(args.gpus)-{4,5,6,7}:
        parser.error('Only physical GPUs 4-7 are authorized')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    state = dict(status='smoke_checks', started_at=now(), controller_pid=os.getpid(), gpus=args.gpus,
        catalog=str(args.catalog.resolve()), catalog_sha256=sha(args.catalog),
        models={model: {'status': 'pending', 'attempts': []} for model in MODELS})
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        state['updated_at'] = now()
        write(output / 'status.json', state)

    def launch(model, stage, gpu, lease, info):
        folder = output / f'{model}_{stage}'
        command = ['bash', str(MODEL_ROOTS[model] / 'local_setup/python_local.sh'),
            str(ROOT / 'local_setup/evaluate_chexchonet.py'), '--model', model,
            '--model-root', str(MODEL_ROOTS[model]), '--catalog', str(args.catalog.resolve()),
            '--output', str(folder), '--batch-size', str(args.batch_size)]
        if stage == 'smoke':
            command += ['--smoke', str(args.batch_size)]
        logpath = output / 'logs' / f'{model}_{stage}.log'
        log = logpath.open('w')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': info['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        try:
            proc = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        except BaseException:
            log.close()
            release_gpu(lease)
            raise
        active[gpu] = (model, stage, proc, lease, log)
        state['models'][model].update(status='running_' + stage, **{stage+'_output': str(folder)})
        state['models'][model]['attempts'].append(dict(stage=stage, pid=proc.pid,
            command=command, output=str(folder), log=str(logpath), gpu=info, started_at=now()))
        print(now(), 'START', model, stage, 'GPU', gpu, 'PID', proc.pid, flush=True)
        save()

    try:
        for stage in ('smoke', 'full'):
            state['status'] = 'smoke_checks' if stage == 'smoke' else 'full_evaluation'
            pending = list(MODELS)
            while pending or active:
                if stopping:
                    raise InterruptedError('Controller stop requested')
                for gpu, (model, active_stage, proc, lease, log) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    worker_status = output / f'{model}_{active_stage}' / 'status.json'
                    passed = code == 0 and worker_status.exists() and json.loads(worker_status.read_text()).get('status') == 'complete'
                    job = state['models'][model]
                    job.update(status=('verified' if stage == 'smoke' else 'complete') if passed else 'failed')
                    job['attempts'][-1].update(returncode=code, finished_at=now())
                    log.close()
                    release_gpu(lease)
                    del active[gpu]
                    print(now(), 'FINISH', model, stage, job['status'], flush=True)
                for model in pending[:]:
                    receipt = ready(model)
                    if receipt is None:
                        continue
                    for gpu in args.gpus:
                        if gpu in active:
                            continue
                        lease, info = reserve_gpu(gpu)
                        if lease is None:
                            state['models'][model]['last_gpu_check'] = info
                            continue
                        state['models'][model]['ready_receipt'] = receipt
                        launch(model, stage, gpu, lease, info)
                        pending.remove(model)
                        break
                save()
                if pending or active:
                    time.sleep(5)
            if any(job['status'] == 'failed' for job in state['models'].values()):
                raise RuntimeError(stage + ' failed; inspect retained logs/results before retry')
        state['status'] = 'finalizing'
        save()
        command = ['bash', str(ROOT.parent / 'AlphaRAD/local_setup/python_local.sh'),
                   str(ROOT / 'local_setup/summarize_chexchonet.py'), '--run', str(output)]
        with (output / 'logs/summary.log').open('w') as log:
            subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='interrupted' if stopping else 'failed', error=repr(exc), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        # Stop only children owned by this controller, never another user's jobs.
        for model, stage, proc, lease, log in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for model, stage, proc, lease, log in active.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            release_gpu(lease)
            log.close()
            state['models'][model]['status'] = 'interrupted'
        state['finished_at'] = now()
        save()
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
