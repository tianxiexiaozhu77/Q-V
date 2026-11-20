"""Validate the project-only runtime and a real CPU inference, without GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=root.parent / 'data/alpharad/ChestXDet10/test_data/36199.png')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if Path(sys.prefix).resolve() != (root / '.venv').resolve():
        raise RuntimeError('Run this with AlphaRAD/.venv/bin/python or local_setup/python_local.sh')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', ALPHARAD_ASSETS=str(root / 'local_assets'))
    checkpoint = root / 'local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    expected = '97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9'
    if digest.hexdigest() != expected:
        raise RuntimeError('Checkpoint does not match the official release SHA256')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], env=env, capture_output=True, text=True)
    (args.output / 'pip_check.log').write_text(check.stdout + check.stderr)
    check.check_returncode()
    inference_outputs = {}
    for label, findings in [('two_findings', ['pleural effusion', 'pneumothorax']), ('one_finding', ['pleural effusion'])]:
        command = [sys.executable, str(root / 'local_setup/infer.py'), '--device', 'cpu', '--threads', '4',
                   '--image', str(args.image), '--checkpoint', str(checkpoint), '--output', str(args.output / label)]
        for finding in findings:
            command.extend(['--finding', finding])
        with (args.output / (label + '.log')).open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        inference_outputs[label] = json.loads((args.output / label / 'result.json').read_text())
    packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'python_executable': sys.executable, 'prefix': sys.prefix,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest.hexdigest(),
        'checkpoint_matches_official': True, 'pip_check': check.stdout.strip(),
        'packages': dict(sorted(packages.items())), 'cpu_inference': inference_outputs,
        'gpu_used': False, 'status': 'complete',
    }
    (args.output / 'environment_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['packages', 'cpu_inference']}, indent=2))


if __name__ == '__main__':
    main()
