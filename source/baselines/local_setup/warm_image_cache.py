#!/usr/bin/env python3
"""Bounded read-only prefetch for NFS evaluation images; no image conversion."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time


def read_one(path):
    size = 0
    with open(path, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
    return size


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    paths = dict.fromkeys(row['image'] for e in json.loads(args.catalog.read_text())['datasets'].values()
                         for row in json.loads(Path(e['manifest']).read_text())['rows'])
    started = time.time()
    total_bytes = done = 0
    # Submit only a small batch at a time. Data is released immediately after
    # read; the kernel can retain cached pages or evict them as memory needs.
    def status(finished=False):
        value = dict(pid=os.getpid(), status='complete' if finished else 'reading',
                     images_read=done, images_total=len(paths), bytes_read=total_bytes,
                     elapsed_seconds=time.time()-started, file_mutations=False,
                     memory_bound='one 1 MiB chunk per worker; optional OS page cache')
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(args.output)
    paths = list(paths)
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        for offset in range(0, len(paths), 128):
            for size in pool.map(read_one, paths[offset:offset+128]):
                total_bytes += size
                done += 1
            status()
    status(True)


if __name__ == '__main__':
    main()
