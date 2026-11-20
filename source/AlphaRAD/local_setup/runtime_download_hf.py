"""Download a pinned public HF snapshot with size and content-hash checks."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def read_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def download_one(repo, revision, root, entry):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(relative)
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('oid') or entry.get('oid')
    algorithm = 'sha256' if entry.get('lfs') else 'sha1'

    def check(path):
        if not path.exists() or path.stat().st_size != entry['size']:
            return None
        digest = hashlib.new(algorithm)
        if algorithm == 'sha1':
            digest.update(('blob %d\0' % entry['size']).encode())
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest() if digest.hexdigest() == expected else None

    if not check(dest):
        part = dest.with_name(dest.name + '.part')
        for attempt in range(5):
            try:
                url = f'https://huggingface.co/{repo}/resolve/{revision}/{entry["path"]}'
                offset = part.stat().st_size if part.exists() else 0
                if offset == entry['size'] and check(part):
                    part.replace(dest)
                    break
                if offset >= entry['size']:
                    offset = 0
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                request = urllib.request.Request(url, headers=headers)
                response = urllib.request.urlopen(request, timeout=90)
                append = offset > 0 and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    response.close()
                    raise ValueError(f'Invalid resume range: {relative}')
                with response, part.open('ab' if append else 'wb') as f:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b''):
                        f.write(chunk)
                if not check(part):
                    raise ValueError(f'Hash/size mismatch: {relative}')
                part.replace(dest)
                break
            except Exception as error:
                print(f'retry {attempt + 1}: {relative}: {type(error).__name__}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    print(f'verified {entry["size"]:>12} {repo}/{relative}', flush=True)
    return dict(path=str(relative), size=entry['size'], hash_type=algorithm, hash=expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repo')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--core-only', action='store_true')
    args = parser.parse_args()
    info = read_json(f'https://huggingface.co/api/models/{args.repo}/revision/{args.revision}')
    revision = info['sha']
    entries = read_json(f'https://huggingface.co/api/models/{args.repo}/tree/{revision}?recursive=true&limit=1000')
    entries = [x for x in entries if x['type'] == 'file']
    if args.core_only:
        entries = [x for x in entries if '/' not in x['path'] and
                   x['path'].endswith(('.json', '.txt', '.safetensors', '.md', '.py'))]
    args.destination.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda x: download_one(args.repo, revision, args.destination, x), entries))
    manifest = dict(repo=args.repo, revision=revision, files=records,
                    total_bytes=sum(x['size'] for x in records))
    (args.destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'COMPLETE: {len(records)} files, {manifest["total_bytes"]} bytes, revision {revision}', flush=True)


if __name__ == '__main__':
    main()
