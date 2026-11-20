#!/usr/bin/env python3
"""Snapshot only unfinished phase12/14 jobs with optional-deadline workers."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from queue_append import atomic_json, verify_sources
from campaign_control import process_identity, owned_group_members

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    state = json.loads((ROOT / 'controller_v2/state.json').read_text())
    if state['status'] != 'stopped':
        raise RuntimeError('Old controller must be stopped before preparing the replacement queue')
    if process_identity(state['pid']) is not None:
        raise RuntimeError('Old controller is still present; verify its exit first')
    stopped = json.loads((ROOT / 'audits/deadline_cancellation_stop.json').read_text())
    for identifier in stopped['active_job_ids']:
        if owned_group_members(state['jobs'][identifier]) != []:
            raise RuntimeError('Old worker group not confirmed gone: ' + identifier)
    target = ROOT / 'queue_no_deadline.json'
    if target.exists():
        raise FileExistsError(target)
    jobs, inherited, replay = [], [], []
    old_queue = json.loads((ROOT / 'queue_v2.json').read_text())['jobs']
    for old in old_queue:
        phase = old['id'].split('_')[0]
        if phase not in ('phase12', 'phase14'):
            continue
        verify_sources(old, required=True)
        record = state['jobs'][old['id']]
        old_output = Path(old['output'])
        new_output = ROOT / 'runs' / (phase + '_no_deadline') / old_output.name
        if new_output.exists() or new_output.is_symlink():
            raise FileExistsError(new_output)
        if record['status'] == 'complete':
            result = json.loads((old_output / 'result.json').read_text())
            assert result['status'] == 'complete' and result['is_smoke'] is False
            assert result['model'] == old['model'] and result['dataset'] == old['dataset']
            inherited.append({'original_job_id': old['id'], 'link': str(new_output),
                              'source': str(old_output), 'result_sha256': sha(old_output / 'result.json')})
            continue
        old_code = ROOT / ('code_' + phase)
        new_code = ROOT / ('code_' + phase + '_no_deadline')
        def remap(text):
            value = str(text)
            if value.startswith(str(old_code) + '/'):
                return str(new_code) + value[len(str(old_code)):]
            return value
        job = copy.deepcopy(old)
        job['id'] = old['id'].replace(phase + '_', phase + '_no_deadline_', 1)
        job['original_job_id'] = old['id']
        job['output'] = str(new_output)
        argv = []
        iterator = iter(old['argv'])
        for arg in iterator:
            if arg == '--deadline':
                next(iterator)
                continue
            argv.append(str(new_output) if arg == str(old_output) else remap(arg))
        job['argv'] = argv
        job['command_marker'] = remap(old['command_marker'])
        job['source_sha256'] = {remap(path): sha(remap(path)) for path in old['source_sha256']}
        verify_sources(job, required=True)
        assert '--deadline' not in argv and job['command_marker'] in argv
        jobs.append(job)
        replay.append({'original_job_id': old['id'], 'new_job_id': job['id'],
                       'old_status': record['status'], 'old_output_preserved': str(old_output),
                       'new_output': str(new_output)})
    assert len(inherited) + len(jobs) == 30, 'Expected exactly two fifteen-task evaluation phases'
    # Do not create links until every old record, source and new command passed preflight.
    for item in inherited:
        link = Path(item['link'])
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(item['source'], target_is_directory=True)
    document = {'created_at': datetime.now(timezone.utc).isoformat(), 'deadline': None,
                'scope': 'Only existing unfinished phase12/phase14 evaluations; no new research.', 'jobs': jobs}
    atomic_json(target, document)
    atomic_json(ROOT / 'audits/no_deadline_queue_manifest.json', {
        'created_at': document['created_at'], 'queue_sha256': sha(target),
        'source_queue_sha256': sha(ROOT / 'queue_v2.json'),
        'source_state_sha256': sha(ROOT / 'controller_v2/state.json'),
        'inherited_complete_results': inherited, 'unfinished_jobs_to_run': replay,
        'deadline': None, 'old_outputs_modified': False,
    })
    print(json.dumps({'new_jobs': len(jobs), 'inherited_complete': len(inherited), 'queue': str(target)}))


if __name__ == '__main__':
    main()
