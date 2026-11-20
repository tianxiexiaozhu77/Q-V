#!/usr/bin/env python3
"""Refresh the finite, user-authorized queue report and publish its final status."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    queue = ROOT / 'queue_no_deadline.json'
    queue_hash = hashlib.sha256(queue.read_bytes()).hexdigest()
    queue_ids = {j['id'] for j in json.loads(queue.read_text())['jobs']}
    with (ROOT / '.finish_current_queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            # This monitor never appends experiments or sends process signals.
            if hashlib.sha256(queue.read_bytes()).hexdigest() != queue_hash:
                raise RuntimeError('Fixed evaluation queue changed; inspect before resuming monitor')
            state = json.loads((ROOT / 'controller_no_deadline/state.json').read_text())
            for script, extra in [
                ('code_no_deadline/campaign_report.py', ['--verify-protected']),
                ('code/unified_result.py', []),
                ('code/summarize_current_queue.py', []),
            ]:
                completed = subprocess.run([sys.executable, str(ROOT / script), *extra],
                                           capture_output=True, text=True)
                print(datetime.now(timezone.utc).isoformat(), script,
                      completed.returncode, completed.stdout, completed.stderr, flush=True)
                if completed.returncode:
                    raise RuntimeError(f'Report generation failed: {script}')
            jobs = state.get('jobs', {})
            all_accepted = queue_ids.issubset(jobs)
            unsettled = {key: jobs.get(key, {}).get('status', 'not_accepted')
                         for key in queue_ids
                         if jobs.get(key, {}).get('status') in (None, 'pending', 'running')}
            if all_accepted and not unsettled and state.get('status') not in ('running', 'starting', 'stopping'):
                failed = {key: jobs[key]['status'] for key in queue_ids
                          if jobs[key]['status'] != 'complete'}
                write_json(ROOT / 'CURRENT_QUEUE_COMPLETE.json', {
                    'finished_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'complete' if not failed else 'finished_with_failures',
                    'deadline': None, 'queue_sha256': queue_hash,
                    'new_jobs': len(queue_ids), 'failed_jobs': failed,
                    'report': str(ROOT / 'CURRENT_QUEUE_RESULTS.txt'),
                    'note': 'Only the existing queue was run; no new research experiments were added.',
                })
                return
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
