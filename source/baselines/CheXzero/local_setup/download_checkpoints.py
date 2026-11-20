"""Download the complete public author ensemble with resumable validated ranges.

Run only one downloader for this checkpoint directory at a time.
"""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = json.loads((ROOT / 'local_setup/checkpoint_listing.json').read_text())

def download(item):
    output = ROOT / 'checkpoints' / item['name']
    if output.exists() and output.stat().st_size == item['size']:
        return
    url = f"https://drive.usercontent.google.com/download?id={item['id']}&export=download&confirm=t"
    subprocess.run([sys.executable, str(ROOT / 'local_setup/download_official_ranges.py'),
                    url, str(output), '--workers', '8'], check=True)
    if output.stat().st_size != item['size']:
        raise RuntimeError(f'Checkpoint size mismatch: {output}')

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, FILES))
    print('All ten official checkpoints downloaded.')
