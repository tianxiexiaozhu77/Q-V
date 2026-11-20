#!/usr/bin/env python3
"""Append immutable job records under the shared queue_v2 flock.

This helper never starts jobs or creates result directories. Every writer must
use this lock: advisory flock cannot protect against an unrelated writer that
ignores it. Identical existing IDs are idempotent, including already-run jobs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


class QueueError(ValueError):
    pass


class QueueConflict(QueueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_json(path):
    def reject(value):
        raise QueueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(job, required=False):
    hashes = job.get("source_sha256", {})
    if not isinstance(hashes, dict) or required and not hashes:
        raise QueueError("Expected nonempty source_sha256 for this job")
    for filename, digest in hashes.items():
        if not isinstance(filename, str) or not Path(filename).is_absolute():
            raise QueueError("Source paths must be absolute")
        if not isinstance(digest, str) or len(digest) != 64:
            raise QueueError("Expected SHA256 source digest")
        if file_sha256(filename) != digest:
            raise QueueConflict(f"Job source changed: {filename}")


def queue_jobs(document):
    jobs = document if isinstance(document, list) else document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, list):
        raise QueueError("Queue must be a job list or an object containing jobs")
    return jobs


def validate_record(job):
    if not isinstance(job, dict):
        raise QueueError("Each job must be an object")
    identifier = job.get("id")
    if not isinstance(identifier, str) or not identifier or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-." for ch in identifier):
        raise QueueError("Unsafe or missing job ID")
    output = job.get("output")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise QueueError("Job output must be an absolute path")
    argv = job.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise QueueError("argv must contain nonempty command arguments")
    canonical(job)  # Reject NaN and unsupported types before any mutation.
    return str(Path(output).resolve())


@contextmanager
def queue_lock(queue_path, guard=None, lock_path=None):
    """A guard is also checked while waiting, so STOP/deadline cannot hang."""
    queue_path = Path(queue_path)
    lock_path = Path(lock_path) if lock_path is not None else queue_path.parent/".queue_v2.append.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    acquired = False
    try:
        while True:
            if guard is not None:
                guard()
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(.05)
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def atomic_json(path, value, guard=None, expected_bytes=None):
    """Fsync data before atomic replacement, then fsync the containing directory."""
    path = Path(path)
    serialized = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.append-", dir=path.parent)
    temporary = Path(filename)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if guard is not None:
            guard()
        if expected_bytes is not None and path.read_bytes() != expected_bytes:
            raise QueueConflict("Queue changed outside the shared append lock; no overwrite performed")
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(serialized).hexdigest()


def append_jobs(queue_path, incoming, *, guard=None, lock_path=None, dry_run=False):
    """Atomically append all new jobs or none; identical IDs are skipped.

    ``guard`` must raise to cancel and is called under the lock and immediately
    before committing. Source hashes are verified for new jobs. Existing queue
    records and their order are never normalized, edited, or removed.
    """
    queue_path = Path(queue_path).resolve()
    if not isinstance(incoming, list):
        raise QueueError("Incoming jobs must be a list")
    incoming = copy.deepcopy(incoming)
    with queue_lock(queue_path, guard=guard, lock_path=lock_path):
        before = queue_path.read_bytes()
        document = read_json(queue_path)
        jobs = queue_jobs(document)
        existing, output_owners = {}, {}
        for job in jobs:
            output = validate_record(job)
            if job["id"] in existing:
                raise QueueConflict(f"Existing queue has duplicate ID: {job['id']}")
            if output in output_owners:
                raise QueueConflict(f"Existing queue has duplicate output: {output}")
            existing[job["id"]] = job
            output_owners[output] = job["id"]
        added, skipped = [], []
        for job in incoming:
            output = validate_record(job)
            identifier = job["id"]
            if identifier in existing:
                if canonical(existing[identifier]) != canonical(job):
                    raise QueueConflict(f"Existing ID differs; create a new ID: {identifier}")
                skipped.append(identifier)
                continue
            if output in output_owners:
                raise QueueConflict(f"Output is already claimed by {output_owners[output]}")
            if os.path.lexists(output):
                raise QueueConflict(f"New job output already exists: {output}")
            verify_sources(job)
            existing[identifier] = job
            output_owners[output] = identifier
            added.append(job)

        def commit_guard():
            if guard is not None:
                guard()
            for job in added:
                if os.path.lexists(str(Path(job["output"]).resolve())):
                    raise QueueConflict(f"New job output appeared before commit: {job['output']}")
                verify_sources(job)

        commit_guard()
        digest = hashlib.sha256(before).hexdigest()
        if added and not dry_run:
            jobs.extend(added)
            digest = atomic_json(queue_path, document, guard=commit_guard, expected_bytes=before)
        return {"status": "dry_run" if dry_run else "appended" if added else "unchanged",
                "added_ids": [job["id"] for job in added], "skipped_ids": skipped,
                "queue_sha256": digest, "new_job_count": len(added),
                "queue_count_after": len(jobs) if not dry_run else len(jobs)+len(added)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT/"queue_v2.json")
    parser.add_argument("--jobs-file", type=Path, required=True,
                        help="JSON job list or object containing jobs; no shell command interpolation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(append_jobs(args.queue, queue_jobs(read_json(args.jobs_file)), dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
