#!/usr/bin/env python3
"""Run the unchanged evaluator with a verified one-batch native pixel cache."""
from pathlib import Path
import shutil

import evaluate_missing as worker
from cached_adapter import Adapter
from query_refinement_common import center_pixels

HERE = Path(__file__).resolve().parent
original_load_query = worker.load_query
original_snapshot = worker.snapshot


def load_query(name, native):
    query = original_load_query(name, native)
    def pixels(paths, view):
        prepared, sizes = native.prepared_pixels(paths)
        return center_pixels(prepared, view), sizes
    query._pixels = pixels
    query.metadata['pixel_cache'] = 'Reuse the current batch of exact native preprocessed pixels; original fit/pair/query/scoring implementations unchanged.'
    return query


def snapshot(args, manifest_sources):
    sources, snapshots, assets = original_snapshot(args, manifest_sources)
    for source in (Path(__file__).resolve(), HERE / 'cached_adapter.py'):
        target = args.output / 'source_snapshot' / source.relative_to(worker.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source)] = worker.sha(source)
        snapshots[str(target.relative_to(args.output))] = worker.sha(target)
        assert worker.sha(target) == sources[str(source)]
    worker.write(args.output / 'source_sha256.json', sources)
    worker.write(args.output / 'snapshot_sha256.json', snapshots)
    return sources, snapshots, assets


worker.Adapter = Adapter
worker.load_query = load_query
worker.snapshot = snapshot

if __name__ == '__main__':
    worker.main()
