#!/usr/bin/env python3
"""Exploratory, fixed-pilot gate that appends full evaluations, never launches GPUs.

Only phase9/phase11/phase13 are eligible. One common candidate must improve all
principal metrics on the same five pilots, with >=1 pp on at least three pairs.
Passing expands ALL candidates to the remaining ten tasks; it never combines
different per-dataset winners. Reused tests are development evidence, not an
independent confirmation. This program runs once unless --watch is requested.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from queue_append import (QueueError, append_jobs, atomic_json, canonical,
                          file_sha256, queue_jobs, read_json, verify_sources)


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_DEADLINE = 1792634400.
ALLOWED_PHASES = ("phase9", "phase11", "phase13")
PILOTS = (("alpharad", "node21"), ("alpharad", "siim_pneumothorax"),
          ("alpharad", "montgomery"), ("radzero", "ChestXDet10"), ("radzero", "SIIM"))
PILOT_PAIRS = frozenset("/".join(pair) for pair in PILOTS)
EXPLORATORY = ("Exploratory gate using repeatedly evaluated test sets; no independent validation claim. "
               "All shared candidates are expanded together; no per-dataset winner selection.")


class GateStopped(RuntimeError):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _argument(argv, flag):
    indices = [i for i, value in enumerate(argv) if value == flag or value.startswith(flag+"=")]
    if len(indices) != 1 or argv[indices[0]] != flag or indices[0]+1 >= len(argv):
        raise QueueError(f"Expected one separate-value {flag} argument")
    return argv[indices[0]+1]


def _replace_argument(argv, flag, value):
    _argument(argv, flag)
    argv[argv.index(flag)+1] = str(value)


def _controller_is_live(root, state):
    """Require the recorded PID to be the live controller for this queue."""
    try:
        pid = state["pid"]
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        proc = Path("/proc")/str(pid)
        fields = (proc/"stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return False
        argv = (proc/"cmdline").read_bytes().decode().strip("\0").split("\0")
        return (str(root/"code/campaign_control.py") in argv
                and Path(_argument(argv, "--queue")).resolve() == root/"queue_v2.json"
                and Path(_argument(argv, "--state-dir")).resolve() == root/"controller_v2")
    except (OSError, ValueError, KeyError, IndexError, UnicodeError):
        return False


class Guard:
    def __init__(self, root, deadline, now=time.time, controller_live=_controller_is_live):
        self.root, self.deadline = Path(root).resolve(), float(deadline)
        self.now, self.controller_live = now, controller_live
        if not math.isfinite(self.deadline) or self.deadline > AUTHORIZED_DEADLINE:
            raise QueueError("Deadline exceeds the authorized 2026-10-22 10:00 Asia/Shanghai cutoff")

    def __call__(self):
        if self.now() >= self.deadline:
            raise GateStopped("authorized deadline reached")
        if (self.root/"STOP_v2").exists():
            raise GateStopped("STOP_v2 exists")
        try:
            state = read_json(self.root/"controller_v2/state.json")
        except (OSError, ValueError) as error:
            raise GateStopped(f"controller state unavailable: {error}") from error
        if state.get("status") != "running":
            raise GateStopped(f"controller terminal/stopping status: {state.get('status')}")
        try:
            controller_deadline = datetime.fromisoformat(state["deadline"].replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError, TypeError) as error:
            raise GateStopped("controller deadline unavailable") from error
        if self.now() >= min(self.deadline, controller_deadline):
            raise GateStopped("controller deadline reached")
        if not self.controller_live(self.root, state):
            raise GateStopped("recorded controller process is no longer live for this queue")
        return state


def load_reporter(root):
    path = Path(root)/"code/campaign_report.py"
    spec = importlib.util.spec_from_file_location("pilot_gate_verified_reporter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task(job):
    return job.get("model"), job.get("dataset")


def _job_id(phase, task):
    return f"{phase}_{task[0]}_{task[1]}"


def baseline_universe(jobs, reporter):
    baseline = [job for job in jobs if job["id"].startswith("phase1_")]
    tasks = [_task(job) for job in baseline]
    if len(baseline) != 15 or len(set(tasks)) != 15 or set(tasks) != set(reporter.TASKS):
        raise QueueError("Phase1 queue must contain exactly the known 15 full tasks")
    for job in baseline:
        if job.get("is_smoke") is not False or job.get("require_result") is not True:
            raise QueueError("Phase1 universe contains smoke or non-result jobs")
    return baseline


def validate_phase_job(root, phase, job, task, deadline):
    if job.get("id") != _job_id(phase, task) or _task(job) != task:
        raise QueueError("Unexpected phase job identity")
    if job.get("is_smoke") is not False or job.get("require_result") is not True:
        raise QueueError("Only explicit full-result phase jobs may be expanded")
    output = root/"runs"/phase/f"{task[0]}_{task[1]}"
    if Path(job["output"]).resolve() != output:
        raise QueueError("Phase output does not match its canonical task path")
    argv = job["argv"]
    if _argument(argv, "--dataset") != task[1] or Path(_argument(argv, "--output")).resolve() != output:
        raise QueueError("Job argv identity/output differs from its record")
    if any(arg == "--candidate-names" or arg.startswith("--candidate-names=") for arg in argv):
        raise QueueError("Pilot expansion must retain all candidates, not a selected winner")
    if any(arg == "--limit-test" or arg.startswith("--limit-test=") for arg in argv):
        if int(_argument(argv, "--limit-test")) != 0:
            raise QueueError("Smoke/truncated pilot cannot be expanded")
    worker = Path(job["command_marker"])
    if str(worker) not in argv or worker.name != f"eval_{task[0]}_candidates.py":
        raise QueueError("Unexpected phase worker")
    candidate = Path(_argument(argv, "--candidate-module"))
    if candidate.name != "query_candidates.py" or str(candidate) not in job.get("source_sha256", {}):
        raise QueueError("Shared candidate source is not pinned in the job")
    if str(worker) not in job.get("source_sha256", {}):
        raise QueueError("Worker source is not pinned in the job")
    if _argument(argv, "--cache-mode") != "read":
        raise QueueError("Full expansion requires the already certified read-only vision cache")
    worker_deadline = float(_argument(argv, "--deadline"))
    if not math.isfinite(worker_deadline) or worker_deadline > min(deadline, AUTHORIZED_DEADLINE):
        raise QueueError("Worker deadline exceeds gate authorization")
    verify_sources(job, required=True)
    return job["source_sha256"][str(candidate)]


def _normalized_template(job):
    value = copy.deepcopy(job)
    value["id"], value["dataset"], value["output"] = "TASK_ID", "DATASET", "OUTPUT"
    _replace_argument(value["argv"], "--dataset", "DATASET")
    _replace_argument(value["argv"], "--output", "OUTPUT")
    return value


def phase_templates(root, phase, jobs, deadline):
    by_id = {job["id"]: job for job in jobs}
    templates, hashes = {}, set()
    for task in PILOTS:
        identifier = _job_id(phase, task)
        if identifier not in by_id:
            return None, None
        job = by_id[identifier]
        hashes.add(validate_phase_job(root, phase, job, task, deadline))
        if task[0] in templates and canonical(_normalized_template(templates[task[0]])) != canonical(_normalized_template(job)):
            raise QueueError("Same-model pilot templates differ beyond dataset/output/ID")
        templates.setdefault(task[0], job)
    if len(hashes) != 1 or set(templates) != {"radzero", "alpharad"}:
        raise QueueError("Pilot jobs do not share one candidate policy and both model templates")
    return templates, hashes.pop()


def eligible_candidates(phase):
    """Use one candidate's FIVE pair minima, never mix winners across pairs."""
    if phase.get("complete_tasks") != 5 or phase.get("failed_or_interrupted"):
        raise QueueError("Pilot report must contain exactly five complete tasks and no failures")
    result = []
    for name, record in phase.get("candidates", {}).items():
        if record.get("complete_tasks") != 5:
            continue
        gains = record.get("per_pair_min_gain_pp", {})
        if set(gains) != PILOT_PAIRS or record.get("main_metric_count") != 9 or record.get("expected_tasks") != 15:
            raise QueueError("Candidate report does not cover the exact five pilot pairs/nine principal metrics")
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                   for value in gains.values()):
            raise QueueError("Nonfinite or invalid pilot gain")
        positive = sum(value > 1e-9 for value in gains.values())
        ge1 = sum(value >= 1-1e-9 for value in gains.values())
        if record.get("all_metric_pairs_positive") != positive or record.get("pairs_all_metrics_ge_1pp") != ge1:
            raise QueueError("Pilot aggregate counts disagree with per-pair minima")
        if positive == 5 and ge1 >= 3:
            result.append({"candidate": name, "pair_min_gains_pp": gains, "positive_pairs": positive, "pairs_ge_1pp": ge1})
    return result


def expansion_jobs(root, phase, baseline, templates):
    expanded = []
    for source in baseline:
        task = _task(source)
        if task in PILOTS:
            continue
        job = copy.deepcopy(templates[task[0]])
        job["id"], job["model"], job["dataset"] = _job_id(phase, task), task[0], task[1]
        job["output"] = str(root/"runs"/phase/f"{task[0]}_{task[1]}")
        _replace_argument(job["argv"], "--dataset", task[1])
        _replace_argument(job["argv"], "--output", job["output"])
        _replace_argument(job["argv"], "--cache-mode", "read")
        expanded.append(job)
    if len(expanded) != 10:
        raise QueueError("Expected exactly ten remaining full tasks")
    return expanded


def inspect_phase(root, phase_name, jobs, baseline, summary, reporter, state, deadline):
    templates, fitter_hash = phase_templates(root, phase_name, jobs, deadline)
    if templates is None:
        return {"status": "waiting", "reason": "fixed pilot jobs are not all queued"}, []
    expected_jobs = expansion_jobs(root, phase_name, baseline, templates)
    by_id = {job["id"]: job for job in jobs}
    if all(job["id"] in by_id for job in expected_jobs):
        for job in expected_jobs:
            if canonical(by_id[job["id"]]) != canonical(job):
                raise QueueError("Existing full phase job differs from the frozen pilot template")
        return {"status": "already_queued", "reason": "all 15 full task IDs already exist unchanged"}, []
    records = state.get("jobs", {})
    evidence_hashes = {}
    for task in PILOTS:
        identifier = _job_id(phase_name, task)
        record = records.get(identifier, {})
        if record.get("status") in ("failed", "stopped", "not_started", "unverified_or_unreaped"):
            return {"status": "rejected", "reason": f"pilot {identifier} is terminal without completion"}, []
        if record.get("status") != "complete":
            return {"status": "waiting", "reason": f"pilot {identifier} has not completed in controller"}, []
        if canonical(record.get("job")) != canonical(by_id[identifier]):
            raise QueueError("Controller pilot record differs from immutable queue job")
        result_path = Path(by_id[identifier]["output"])/"result.json"
        baseline_job = next(job for job in baseline if _task(job) == task)
        baseline_path = Path(baseline_job["output"])/"result.json"
        evidence_paths = [result_path, baseline_path]
        if task[0] == "alpharad":
            evidence_paths.append(result_path.parent/"data_audit.json")
        for path in evidence_paths:
            evidence_hashes[str(path)] = file_sha256(path)
        result, baseline_result = read_json(result_path), read_json(baseline_path)
        if (result.get("status") != "complete" or result.get("is_smoke") is not False
                or (result.get("model"), result.get("dataset")) != task
                or result.get("task") != reporter.TASKS[task]
                or result.get("rows") != baseline_result.get("rows")
                or result.get("support_count") != baseline_result.get("support_count")):
            raise QueueError("Pilot result is failed, smoke, truncated, or uses different support/task identity")
    phases = [value for value in summary.get("phases", []) if value.get("phase") == phase_name]
    if len(phases) != 1:
        return {"status": "waiting", "reason": "verified summary has not reported this complete pilot phase"}, []
    reported = phases[0]
    # Recompute only the reporter's pure verification/aggregation function.
    # This neither edits reports nor allows an outdated summary to authorize
    # a now-failed/smoke/mixed-source result set.
    current, _ = reporter.summarize_phase(root/"runs"/phase_name)
    if canonical(current) != canonical(reported):
        return {"status": "waiting", "reason": "summary is not current for the validated result files"}, []
    if reported.get("shared_fitter_sha256") != fitter_hash:
        raise QueueError("Reported candidate source does not equal frozen pilot source")
    if any(file_sha256(path) != digest for path, digest in evidence_hashes.items()):
        raise QueueError("Pilot evidence changed during verification")
    passing = eligible_candidates(reported)
    if not passing:
        return {"status": "rejected", "reason": "no single common candidate passes five-positive/three-ge1pp gate",
                "shared_fitter_sha256": fitter_hash}, []
    return {"status": "eligible", "passing_shared_candidates": passing,
            "shared_fitter_sha256": fitter_hash,
            "evidence_files_sha256": evidence_hashes,
            "report_phase_sha256": hashlib.sha256(canonical(reported)).hexdigest(),
            "phase_result_paths": sorted(str(path) for path in (root/"runs"/phase_name).glob("*/result.json")),
            "expansion_policy": "all candidates, ten remaining tasks, same model-specific frozen template"}, expected_jobs


def run_once(root=ROOT, phases=ALLOWED_PHASES, deadline=AUTHORIZED_DEADLINE, *,
             now=time.time, controller_live=_controller_is_live, reporter=None,
             persist=True, dry_run=False):
    root = Path(root).resolve()
    phases = list(phases)
    if not phases or len(set(phases)) != len(phases) or not set(phases).issubset(ALLOWED_PHASES):
        raise QueueError("Gate config permits only unique phase9, phase11, phase13 entries")
    guard = Guard(root, deadline, now=now, controller_live=controller_live)
    output = {"updated_at": now_iso(), "status": "running", "interpretation": EXPLORATORY,
              "deadline_epoch": deadline, "configured_phases": phases, "phases": {}}
    try:
        state = guard()
        summary_bytes = (root/"summary.json").read_bytes()
        summary = json.loads(summary_bytes)
        if summary.get("expected_tasks") != 15 or summary.get("expected_pairs") != 14:
            raise QueueError("Summary does not represent the complete 15-task/14-pair campaign")
        output["summary_sha256"] = hashlib.sha256(summary_bytes).hexdigest()
        reporter = load_reporter(root) if reporter is None else reporter
        jobs = queue_jobs(read_json(root/"queue_v2.json"))
        baseline = baseline_universe(jobs, reporter)
        for phase in phases:
            state = guard()
            try:
                decision, new_jobs = inspect_phase(root, phase, jobs, baseline, summary, reporter, state, deadline)
                if new_jobs:
                    pilot_jobs = {_job_id(phase, task): next(job for job in jobs if job["id"] == _job_id(phase, task))
                                  for task in PILOTS}
                    def evidence_guard():
                        live = guard()
                        for identifier, job in pilot_jobs.items():
                            record = live.get("jobs", {}).get(identifier, {})
                            if record.get("status") != "complete" or canonical(record.get("job")) != canonical(job):
                                raise QueueError("Pilot completion changed before queue commit")
                        for path, digest in decision["evidence_files_sha256"].items():
                            if file_sha256(path) != digest:
                                raise QueueError("Pilot evidence changed before queue commit")
                        paths = sorted(str(path) for path in (root/"runs"/phase).glob("*/result.json"))
                        if paths != decision["phase_result_paths"]:
                            raise QueueError("Phase result set changed before queue commit")
                        latest = [entry for entry in read_json(root/"summary.json").get("phases", []) if entry.get("phase") == phase]
                        if len(latest) != 1 or hashlib.sha256(canonical(latest[0])).hexdigest() != decision["report_phase_sha256"]:
                            raise QueueError("Verified pilot summary changed before queue commit")
                    result = append_jobs(root/"queue_v2.json", new_jobs, guard=evidence_guard, dry_run=dry_run)
                    decision["append_result"] = result
                    decision["status"] = "eligible_dry_run" if dry_run else "expanded"
                    jobs = queue_jobs(read_json(root/"queue_v2.json"))
                output["phases"][phase] = decision
            except (OSError, ValueError, KeyError, TypeError) as error:
                output["phases"][phase] = {"status": "error", "reason": str(error)}
        terminal = {"rejected", "expanded", "already_queued", "error", "eligible_dry_run"}
        if any(record["status"] == "error" for record in output["phases"].values()):
            output["status"] = "error"
        elif all(record["status"] in terminal for record in output["phases"].values()):
            output["status"] = "finished"
        guard()
    except GateStopped as error:
        output.update(status="stopped", reason=str(error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        output.update(status="error", reason=str(error))
    if persist:
        atomic_json(root/"pilot_gate_state.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, help='Optional JSON object containing only "phases"')
    parser.add_argument("--deadline", type=float, default=AUTHORIZED_DEADLINE)
    parser.add_argument("--watch", action="store_true", help="Explicitly opt into polling; never launches workers itself")
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    phases = ALLOWED_PHASES
    if args.config:
        config = read_json(args.config)
        if not isinstance(config, dict) or set(config) != {"phases"}:
            parser.error('Config must contain exactly "phases"')
        phases = config["phases"]
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    while True:
        result = run_once(args.root, phases, args.deadline, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch or result["status"] != "running":
            return 1 if result["status"] == "error" else 0
        wake = time.monotonic()+args.poll_seconds
        guard = Guard(args.root, args.deadline)
        while time.monotonic() < wake:
            try:
                guard()
            except GateStopped:
                stopped = run_once(args.root, phases, args.deadline, dry_run=args.dry_run)
                print(json.dumps(stopped, ensure_ascii=False), flush=True)
                return 0
            time.sleep(min(1., max(0., wake-time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
