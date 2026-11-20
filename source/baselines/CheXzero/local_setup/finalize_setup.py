"""Run native CPU parity, hash assets, and seal a completed CheXzero installation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'local_setup'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if (SETUP / 'READY.json').exists():
        raise RuntimeError('Setup is already sealed; do not overwrite a running evaluation protocol')
    listing = json.loads((SETUP / 'checkpoint_listing.json').read_text())
    checkpoints = []
    for item in listing:
        path = ROOT / 'checkpoints' / item['name']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError(f'Checkpoint is missing or incomplete: {path}')
        checkpoints.append(dict(path=str(path), size=path.stat().st_size,
                                sha256=digest(path), google_drive_id=item['id'],
                                official_checksum_available=False))
    print('All ten checkpoint sizes verified and SHA256 recorded.', flush=True)
    image = json.loads((ROOT.parent / 'BiomedCLIP/local_setup/cpu_smoke.json').read_text())['image']
    with (SETUP / 'cpu_smoke.log').open('w') as log:
        subprocess.run(['bash', str(SETUP / 'python_local.sh'), str(SETUP / 'cpu_smoke.py'), image],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    smoke_path = SETUP / 'cpu_smoke.json'
    smoke = json.loads(smoke_path.read_text())
    if smoke['status'] != 'passed' or smoke['metadata']['checkpoint_count'] != 10:
        raise RuntimeError('Full-ensemble CPU validation did not pass')
    validation = dict(path=str(smoke_path), sha256=digest(smoke_path), status='passed',
                      native_ensemble_max_abs_diff=smoke['native_ensemble_max_abs_diff'])
    adapter_hash = digest(SETUP / 'adapter.py')
    launcher_hash = digest(SETUP / 'python_local.sh')
    manifest_path = SETUP / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status='ready', checkpoints=checkpoints, cpu_validation=validation,
                    adapter_sha256=adapter_hash,
                    launcher_sha256=launcher_hash,
                    evaluation_worker=str(ROOT.parent / 'local_setup/evaluate_baseline_buffered.py'),
                    evaluation_worker_sha256=digest(ROOT.parent / 'local_setup/evaluate_baseline_buffered.py'),
                    requirements_lock_sha256=digest(SETUP / 'requirements.lock.txt'),
                    inference_metadata=smoke['metadata'])
    manifest['notes'] = [
        'No training. Complete official ensemble of all ten public author checkpoints; filenames are not selected using the current test labels.',
        'Exact expected byte sizes and HTTP Content-Range validated; SHA256 recorded after download. Google Drive listing does not publish a separate checksum.',
        'Native CheXzero binary positive/negative raw-cosine softmax, then arithmetic mean across ten model probabilities. No logit scaling.',
        'AlphaRAD common positive prompt There is {disease}.; negative adapted consistently as There is no {disease}. AlphaRAD paper does not specify the negative-template details, so this is not claimed to exactly reproduce its original table.',
        'Pleural_Thickening metadata identifier becomes pleural thickening in text prompts, as explicitly defined by AlphaRAD disease_name_mapping; output columns remain unchanged.',
        'Native authors pixel-domain mean/std and long-edge 320 grayscale padding retained, followed by tensor bicubic 224 Resize with antialias=False, matching author torchvision version.',
        'No verified native localization/segmentation interface is claimed.'
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    ready = dict(ready=True, model='CheXzero', adapter_sha256=adapter_hash,
                 launcher_sha256=launcher_hash,
                 manifest_sha256=digest(manifest_path), cpu_validation=validation,
                 created_at=datetime.now(timezone.utc).isoformat())
    (SETUP / 'READY.json').write_text(json.dumps(ready, indent=2) + '\n')
    print(json.dumps(ready, indent=2), flush=True)


if __name__ == '__main__':
    main()
