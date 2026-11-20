#!/usr/bin/env python3
"""Run frozen CheXchoNet comparisons, after GPU leases and native-API smoke checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'AlphaRAD/local_setup'))
from run_pretrained import reserve_gpu, release_gpu


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'AlphaRAD/local_setup/chexchonet_20261023/catalog.json')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    state = {'status': 'smoke_checks', 'started_at': now(), 'controller_pid': os.getpid(),
             'gpus': [4, 5, 6, 7], 'catalog': str(args.catalog.resolve()), 'jobs': {}}
    leases, active = {}, {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def save():
        write(out / 'status.json', state)

    def launch(name, model, mode, gpu, smoke=False):
        project = ROOT / ('AlphaRAD' if model == 'alpharad' else 'RadZero_i2cl')
        command = ['bash', str(project / 'local_setup/python_local.sh'),
                   str(project / 'local_setup/eval_chexchonet.py'), '--catalog', str(args.catalog.resolve()),
                   '--output', str(out / name), '--mode', mode, '--batch-size', str(args.batch_size),
                   '--workers', str(args.workers)]
        if smoke:
            command += ['--smoke-batches', '1'] if model == 'alpharad' else ['--limit', str(args.batch_size)]
        lease, info = reserve_gpu(gpu)
        if lease is None:
            raise RuntimeError(f'GPU {gpu} is unavailable: {info}')
        leases[gpu] = lease
        log = (out / 'logs' / f'{name}.log').open('w')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': info['uuid'], 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
               'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'OPENBLAS_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        proc = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
        active[name] = (proc, log, gpu)
        state['jobs'][name] = {'status': 'running', 'pid': proc.pid, 'gpu': info, 'mode': mode,
                               'smoke': smoke, 'command': command, 'started_at': now()}
        save()
        print(f'{now()} Started {name}: GPU {gpu}, PID {proc.pid}', flush=True)

    def wait_phase():
        while active and not stopping:
            for name, (proc, log, gpu) in list(active.items()):
                rc = proc.poll()
                if rc is not None:
                    result_file = out / name / ('smoke_result.json' if name == 'alpharad_smoke' else 'result.json')
                    passed = rc == 0 and result_file.is_file()
                    state['jobs'][name].update(status='complete' if passed else 'failed', exit_code=rc, finished_at=now())
                    log.close()
                    release_gpu(leases.pop(gpu))
                    del active[name]
                    print(f'{now()} Finished {name}: {state["jobs"][name]["status"]}', flush=True)
                    save()
            if active:
                time.sleep(5)
        if stopping:
            raise InterruptedError('Controller stop requested')
        if any(j['status'] == 'failed' for j in state['jobs'].values()):
            raise RuntimeError('One or more jobs failed; inspect logs/failure.json')

    try:
        save()
        launch('alpharad_smoke', 'alpharad', 'paired-four-arm', 4, True)
        launch('radzero_smoke', 'radzero', 'paired-four-arm', 5, True)
        wait_phase()
        state['status'] = 'full_evaluation'
        launch('alpharad_full', 'alpharad', 'full-original', 4)
        launch('radzero_full', 'radzero', 'full-original', 5)
        launch('alpharad_paired', 'alpharad', 'paired-four-arm', 6)
        launch('radzero_paired', 'radzero', 'paired-four-arm', 7)
        wait_phase()
        state['status'] = 'summarizing'
        save()
        command = ['bash', str(ROOT / 'AlphaRAD/local_setup/python_local.sh'),
                   str(Path(__file__).with_name('summarize.py')), '--run', str(out)]
        with (out / 'logs/summary.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='interrupted' if stopping else 'failed', error=repr(exc))
        print(repr(exc), flush=True)
    finally:
        for proc, log, gpu in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for name, (proc, log, gpu) in active.items():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
            state['jobs'][name].update(status='interrupted', finished_at=now())
        for lease in leases.values():
            release_gpu(lease)
        state['finished_at'] = now()
        save()
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
