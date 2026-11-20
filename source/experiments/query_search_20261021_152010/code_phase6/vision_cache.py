"""Immutable, ordered FP32 visual-token shards with audited completion.

The caller owns model/preprocessing identity and all control computations. This
module never considers an interrupted prefix a reusable, certified full cache.
Only the unchanged and eta=.3 controls passing on every image certify a cache.
No torch dependency; caches must be placed on the experiment storage volume.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import time
import uuid

import numpy as np

VERSION = 'ordered-fp32-visual-tokens-v1'


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def image_fingerprints(paths):
    """Ordered paths plus current file stat identity; not content hashes.

    Callers should additionally include existing content manifests in identity.
    Full prediction controls remain mandatory even when all stats match.
    """
    result = []
    for name in paths:
        path = Path(name)
        stat = path.stat()
        result.append({'path': str(path.resolve()), 'bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns})
    return result


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class VisionTokenCache:
    def __init__(self, root, identity, image_ids, mode='auto', shard_rows=64,
                 prefetch=1, deadline=None):
        if mode not in ('auto', 'build', 'read'):
            raise ValueError('Instantiate cache only for auto/build/read; off is caller-owned')
        if shard_rows < 1 or prefetch not in (0, 1):
            raise ValueError('Positive shard_rows and prefetch 0 or 1 required')
        if not image_ids or len(set(image_ids)) != len(image_ids):
            raise ValueError('Expected nonempty unique ordered image IDs')
        self.identity = json.loads(canonical_json(identity))
        self.image_ids = list(image_ids)
        self.key = hashlib.sha256(canonical_json({'version': VERSION, 'identity': identity,
                                                  'image_ids': image_ids})).hexdigest()
        self.path = Path(root).resolve() / self.key
        self.manifest_path = self.path / 'manifest.json'
        self.partial_path = self.path / 'partial.json'
        self.lock_path = self.path / 'writer.lock'
        self.deadline, self.shard_rows = deadline, int(shard_rows)
        self._lock_token = None
        self._buffer, self._buffer_rows = [], 0
        self._memory = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self._future = None
        self._closed = False
        self.shape = None
        self.cached_rows = self.written_rows = 0
        self._check_deadline()
        self.path.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.mode = 'read'
            self.manifest = self._load_manifest(self.manifest_path, complete=True)
        elif mode == 'read':
            self.close()
            raise FileNotFoundError(f'No certified complete visual cache: {self.manifest_path}')
        else:
            self.mode = 'build'
            self._acquire_lock()
            try:
                # Another writer may have completed before this lock was acquired.
                if self.manifest_path.exists():
                    self.mode = 'read'
                    self.manifest = self._load_manifest(self.manifest_path, complete=True)
                    self._release_lock()
                elif self.partial_path.exists():
                    self.manifest = self._load_manifest(self.partial_path, complete=False)
                else:
                    self.manifest = {'version': VERSION, 'key': self.key, 'status': 'building',
                        'identity': self.identity, 'image_ids': self.image_ids, 'rows': len(self.image_ids),
                        'dtype': 'float32', 'token_shape': None, 'shards': [], 'cached_rows': 0,
                        'created_epoch': time.time()}
                    _atomic_json(self.partial_path, self.manifest)
            except BaseException:
                self.close()
                raise
        self.cached_rows = self.written_rows = self.manifest['cached_rows']
        if self.manifest['token_shape'] is not None:
            self.shape = tuple(self.manifest['token_shape'])

    def _check_deadline(self):
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError('Visual cache deadline reached')

    def _acquire_lock(self):
        token = uuid.uuid4().hex
        try:
            self.lock_path.mkdir()
        except FileExistsError:
            owner_path = self.lock_path / 'owner.json'
            owner = json.loads(owner_path.read_text()) if owner_path.exists() else {}
            # Only recover a demonstrably dead process on this same machine.
            if owner.get('host') == socket.gethostname() and isinstance(owner.get('pid'), int):
                try:
                    os.kill(owner['pid'], 0)
                except ProcessLookupError:
                    stale = self.path / ('dead-writer-' + uuid.uuid4().hex)
                    os.rename(self.lock_path, stale)
                    (stale / 'owner.json').unlink(missing_ok=True)
                    stale.rmdir()
                    return self._acquire_lock()
            raise RuntimeError(f'Visual cache already has a writer: {owner}')
        self._lock_token = token
        _atomic_json(self.lock_path / 'owner.json', {'pid': os.getpid(), 'host': socket.gethostname(),
                                                   'token': token, 'started_epoch': time.time()})

    def _release_lock(self):
        if self._lock_token is None:
            return
        owner_path = self.lock_path / 'owner.json'
        owner = json.loads(owner_path.read_text())
        if owner.get('token') != self._lock_token:
            raise RuntimeError('Visual cache writer lock ownership changed')
        owner_path.unlink()
        self.lock_path.rmdir()
        self._lock_token = None

    def _load_manifest(self, path, complete):
        record = json.loads(path.read_text())
        if (record.get('version') != VERSION or record.get('key') != self.key or
            record.get('identity') != self.identity or record.get('image_ids') != self.image_ids or
            record.get('rows') != len(self.image_ids) or record.get('dtype') != 'float32'):
            raise ValueError('Visual cache identity/order/version mismatch')
        if complete:
            if record.get('status') != 'complete' or record.get('cached_rows') != len(self.image_ids):
                raise ValueError('Visual cache is not complete')
            self._validate_controls(record.get('controls', {}))
        cursor = 0
        for shard in record['shards']:
            filename = shard['file']
            if Path(filename).name != filename or not filename.endswith('.npy'):
                raise ValueError('Unsafe cache shard path')
            if shard['start'] != cursor or not cursor < shard['end'] <= len(self.image_ids):
                raise ValueError('Noncontiguous visual cache shards')
            if (self.path / filename).stat().st_size != shard['bytes']:
                raise ValueError('Cache shard file size changed')
            cursor = shard['end']
        if cursor != record['cached_rows']:
            raise ValueError('Visual cache prefix coverage mismatch')
        return record

    def _validate_controls(self, controls):
        for name in ('original', 'uniform_eta03'):
            value = controls.get(name, {})
            if value.get('passed') is not True or value.get('images_checked') != len(self.image_ids):
                raise ValueError(f'Cannot certify cache: {name} must pass on all {len(self.image_ids)} images')

    def _read_shard(self, index):
        self._check_deadline()
        record = self.manifest['shards'][index]
        data = (self.path / record['file']).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'Visual cache shard checksum mismatch: {record["file"]}')
        array = np.load(io.BytesIO(data), allow_pickle=False)
        expected_shape = (record['end'] - record['start'], *self.shape)
        if array.dtype != np.float32 or array.shape != expected_shape or not np.isfinite(array).all():
            raise ValueError('Invalid visual cache dtype/shape/values')
        array.setflags(write=False)
        self._check_deadline()
        return array

    def _array(self, index):
        if index in self._memory:
            self._memory.move_to_end(index)
            value = self._memory[index]
        elif self._future is not None and self._future[0] == index:
            value = self._future[1].result()
            self._future = None
            self._memory[index] = value
        else:
            value = self._read_shard(index)
            self._memory[index] = value
        while len(self._memory) > 2:
            self._memory.popitem(last=False)
        if self._executor and self._future is None and index + 1 < len(self.manifest['shards']):
            if index + 1 not in self._memory:
                self._future = (index + 1, self._executor.submit(self._read_shard, index + 1))
        return value

    def get(self, start, end):
        self._check_deadline()
        if not 0 <= start < end <= len(self.image_ids):
            raise ValueError('Invalid visual cache row slice')
        if end > self.cached_rows:
            return None
        parts = []
        for index, shard in enumerate(self.manifest['shards']):
            if shard['end'] <= start:
                continue
            if shard['start'] >= end:
                break
            array = self._array(index)
            parts.append(array[max(start, shard['start'])-shard['start']:min(end, shard['end'])-shard['start']])
        result = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if len(result) != end - start:
            raise ValueError('Cache returned incomplete row slice')
        return result

    def append(self, start, tokens_fp32):
        self._check_deadline()
        if self.mode != 'build' or self._closed or self._lock_token is None:
            raise RuntimeError('Cache is not open for writing')
        array = np.asarray(tokens_fp32)
        if array.dtype != np.float32 or array.ndim != 3 or not len(array) or not np.isfinite(array).all():
            raise ValueError('Cache accepts finite nonempty FP32 [B,T,D] arrays only')
        if start != self.written_rows or start + len(array) > len(self.image_ids):
            raise ValueError(f'Cache append must start at row {self.written_rows}, got {start}')
        if self.shape is None:
            self.shape = tuple(array.shape[1:])
            self.manifest['token_shape'] = list(self.shape)
        if tuple(array.shape[1:]) != self.shape:
            raise ValueError('Visual token shape changed')
        self._buffer.append(np.array(array, copy=True, order='C'))
        self._buffer_rows += len(array)
        self.written_rows += len(array)
        # Flush whole batches so restart prefixes preserve caller batch grouping.
        if self._buffer_rows >= self.shard_rows:
            self._flush()

    def _flush(self):
        self._check_deadline()
        if not self._buffer_rows:
            return
        value = self._buffer[0] if len(self._buffer) == 1 else np.concatenate(self._buffer)
        memory = io.BytesIO()
        np.save(memory, value, allow_pickle=False)
        payload = memory.getbuffer()
        filename = f'{self.cached_rows:08d}-{self.written_rows:08d}-{uuid.uuid4().hex}.npy'
        target = self.path / filename
        temporary = target.with_suffix('.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            record = {'file': filename, 'start': self.cached_rows, 'end': self.written_rows,
                      'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
            self.manifest['shards'].append(record)
            self.manifest['cached_rows'] = self.written_rows
            self.manifest['updated_epoch'] = time.time()
            _atomic_json(self.partial_path, self.manifest)
            self.cached_rows = self.written_rows
            self._buffer, self._buffer_rows = [], 0
        finally:
            payload.release()
            memory.close()
            temporary.unlink(missing_ok=True)

    def finalize(self, controls, extra=None):
        self._check_deadline()
        self._validate_controls(controls)
        if self.written_rows != len(self.image_ids):
            raise ValueError('Cannot certify incomplete cache')
        if self.mode == 'read':
            return self.manifest
        self._flush()
        self.manifest.update(status='complete', controls=controls, extra=extra or {},
                             completed_epoch=time.time())
        _atomic_json(self.manifest_path, self.manifest)
        self.mode = 'read'
        self._release_lock()
        return self.manifest

    def abort(self, reason):
        # Buffered, uncommitted rows are intentionally discarded; callers replay
        # from the persisted batch boundary and recheck both controls on resume.
        if self.mode == 'build' and self._lock_token is not None:
            self.manifest.update(status='interrupted', reason=str(reason), updated_epoch=time.time())
            _atomic_json(self.partial_path, self.manifest)
        self._buffer, self._buffer_rows = [], 0
        self.written_rows = self.cached_rows
        self.close()

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._release_lock()
        self._memory.clear()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.abort(exc if exc is not None else 'context closed before certification')
