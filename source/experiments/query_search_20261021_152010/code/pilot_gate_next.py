#!/usr/bin/env python3
"""One-shot exploratory pilot gate; dry-run unless --commit is explicit.

Only phases 12/14/15/16/17/17b/18 are accepted. A single non-control candidate must
improve every principal metric on all five fixed pilots, and improve every
metric by >=1 pp on at least three pilots. This is only a necessary condition
for the 12-of-14 pair target: ten remaining tasks represent nine further pairs.
All candidates are retained for expansion. Repeated-test selection is exploratory.
This module never launches a worker, GPU process, controller or watcher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import time

from pilot_gate import (AUTHORIZED_DEADLINE, ROOT, PILOTS, PILOT_PAIRS, Guard,
    GateStopped, _argument, _replace_argument, _controller_is_live, _job_id,
    _task, load_reporter, now_iso, validate_phase_job)
from queue_append import (QueueError, append_jobs, atomic_json, canonical,
                          file_sha256, queue_jobs, read_json, verify_sources)

ALLOWED_PHASES = ("phase12", "phase14", "phase15", "phase16", "phase17", "phase17b", "phase18")
CONTROLS = ("original", "uniform_eta03")
EXPLORATORY = ("Fixed support-only candidate fitting; campaign-level selection repeatedly uses test sets "
    "and is exploratory, not independent validation. One shared candidate must pass all five pilots; "
    "no per-dataset winner stitching. All candidates are expanded together.")


def _unique_jobs(jobs):
    result = {}
    for job in jobs:
        if job["id"] in result:
            raise QueueError("Duplicate queue job ID: " + job["id"])
        result[job["id"]] = job
    return result


def phase6_universe(root, jobs, reporter, deadline):
    baseline = [job for job in jobs if job["id"].startswith("phase6_")]
    if len(baseline) != 15 or {_task(job) for job in baseline} != set(reporter.TASKS):
        raise QueueError("Phase6 must contain exactly the known 15 full task templates")
    model_templates = {}
    for job in baseline:
        validate_phase_job(root, "phase6", job, _task(job), deadline)
        normalized = canonical(_normalized_template(job))
        if job["model"] in model_templates and model_templates[job["model"]] != normalized:
            raise QueueError("Phase6 model templates contain unhandled task-specific CLI differences")
        model_templates[job["model"]] = normalized
    return baseline


def _normalized_template(job):
    """Dataset/output placeholders; dataset-specific feature paths stay explicit."""
    result = copy.deepcopy(job)
    result.update(id="TASK_ID", dataset="DATASET", output="OUTPUT")
    _replace_argument(result["argv"], "--dataset", "DATASET")
    _replace_argument(result["argv"], "--output", "OUTPUT")
    if "--global-support-features" in result["argv"]:
        # Never transplant ChestXDet10's global support path into another dataset.
        _replace_argument(result["argv"], "--global-support-features", "DATASET_FEATURES")
    return result


def frozen_templates(root, phase, jobs, deadline):
    path = root/"audits"/(phase+"_freeze.json")
    if not path.exists():
        return None
    audit = read_json(path)
    if audit.get("status") != "complete":
        raise QueueError("Phase freeze audit is not complete")
    frozen = audit.get("jobs", [])
    if len(frozen) != 5 or {_task(job) for job in frozen} != set(PILOTS):
        raise QueueError("Freeze audit must contain exactly the five fixed pilot jobs")
    verify_sources({"source_sha256": audit.get("source_sha256")}, required=True)
    by_id, templates, fitter_hashes = _unique_jobs(jobs), {}, set()
    for job in frozen:
        task = _task(job)
        fitter_hashes.add(validate_phase_job(root, phase, job, task, deadline))
        expected_worker = root/("code_"+phase)/f"eval_{task[0]}_candidates.py"
        if Path(job["command_marker"]).resolve() != expected_worker:
            raise QueueError("Freeze worker is outside its actual phase code directory")
        candidate = Path(_argument(job["argv"], "--candidate-module")).resolve()
        expected_candidate = root/("code_"+phase)/"query_candidates.py"
        if phase in ("phase17", "phase17b"):
            expected_candidate = root/"code_multiview_prototype/query_candidates.py"
            if audit.get("canonical_candidate_module") != str(expected_candidate):
                raise QueueError("Canonical multiview candidate exception is not declared by its freeze audit")
        if candidate != expected_candidate:
            raise QueueError("Frozen candidate path differs from the authorized phase policy")
        if canonical(job["source_sha256"]) != canonical(audit["source_sha256"]):
            raise QueueError("Frozen pilot source pins differ from the phase audit")
        actual = by_id.get(job["id"])
        if actual is None:
            return None
        if canonical(actual) != canonical(job):
            raise QueueError("Queue pilot differs from its frozen audit template")
        if task[0] in templates and canonical(_normalized_template(templates[task[0]])) != canonical(_normalized_template(job)):
            raise QueueError("Same-model frozen pilot CLI differs beyond task identity")
        templates.setdefault(task[0], job)
    if len(fitter_hashes) != 1 or set(templates) != {"radzero", "alpharad"}:
        raise QueueError("Five pilots do not share the same candidate source and both native workers")
    return {"path": path, "sha256": file_sha256(path), "audit": audit,
            "templates": templates, "fitter_hash": fitter_hashes.pop(), "jobs": frozen}


def expansion_jobs(root, phase, baseline, frozen):
    """Use phase6's exact task order; use the actual phase's frozen worker/CLI.

    An optional global-support input is resolved from the corresponding phase6
    task, only when the phase's frozen worker still requests that input. Other
    CLI arguments, source pins, launcher and candidate-module remain unchanged.
    """
    for pilot in frozen["jobs"]:
        if "--global-support-features" in pilot["argv"]:
            reference = next(job for job in baseline if _task(job) == _task(pilot))
            if (_argument(pilot["argv"], "--global-support-features") !=
                    _argument(reference["argv"], "--global-support-features")):
                raise QueueError("Frozen global support input is not the corresponding phase6 input")
    result = []
    for source in baseline:
        task = _task(source)
        if task in PILOTS:
            continue
        job = copy.deepcopy(frozen["templates"][task[0]])
        job.update(id=_job_id(phase, task), model=task[0], dataset=task[1],
                   output=str(root/"runs"/phase/f"{task[0]}_{task[1]}"))
        _replace_argument(job["argv"], "--dataset", task[1])
        _replace_argument(job["argv"], "--output", job["output"])
        if "--global-support-features" in job["argv"]:
            _replace_argument(job["argv"], "--global-support-features",
                              _argument(source["argv"], "--global-support-features"))
        result.append(job)
    pairs = {(m, "chestxdet10" if m == "alpharad" and d == "chestxdet10_gt" else d)
             for m, d in map(_task, result)}
    if len(result) != 10 or len(pairs) != 9:
        raise QueueError("Expansion must contain ten tasks representing nine remaining pairs")
    return result


def candidate_evidence(phase):
    if phase.get("complete_tasks") != 5 or phase.get("failed_or_interrupted"):
        raise QueueError("Pilot summary must have exactly five complete tasks and no failed/interrupted results")
    evidence, passing = {}, []
    for name, record in phase.get("candidates", {}).items():
        item = {"candidate": name, "reported": copy.deepcopy(record), "eligible": False}
        evidence[name] = item
        if record.get("complete_tasks") != 5:
            item["reason"] = "candidate does not cover all five pilots"
            continue
        gains = record.get("per_pair_min_gain_pp", {})
        if set(gains) != PILOT_PAIRS or record.get("main_metric_count") != 9 or record.get("expected_tasks") != 15:
            raise QueueError("Candidate does not cover the five fixed pairs and nine principal metrics")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in gains.values()):
            raise QueueError("Nonfinite/invalid pilot gain")
        positive = sum(v > 1e-9 for v in gains.values())
        ge1 = sum(v >= 1-1e-9 for v in gains.values())
        if record.get("all_metric_pairs_positive") != positive or record.get("pairs_all_metrics_ge_1pp") != ge1:
            raise QueueError("Reported pilot aggregates disagree with per-pair minima")
        item.update(pair_min_gains_pp=copy.deepcopy(gains), positive_pairs=positive,
                    pairs_ge_1pp=ge1, remaining_pairs=9, maximum_possible_pairs_ge_1pp=ge1+9)
        if name in CONTROLS:
            item["reason"] = "historical control is excluded from gate eligibility"
        elif positive == 5 and ge1 >= 3:
            item.update(eligible=True, reason="same candidate passes all five pilots and at least three >=1pp")
            passing.append(name)
        else:
            item["reason"] = "requires five positive pairs and at least three pairs with every metric >=1pp"
    if not all(name in evidence for name in CONTROLS):
        raise QueueError("Two historical controls must be present in the pilot report")
    return evidence, passing


def _result_provenance(result, path, model):
    return result.get("config", {}) if model == "radzero" else read_json(path.parent/"data_audit.json")


def inspect_phase(root, phase, jobs, baseline, summary, reporter, state, deadline):
    frozen = frozen_templates(root, phase, jobs, deadline)
    if frozen is None:
        return {"status": "waiting", "reason": "complete freeze audit and exact five queued pilots required"}, []
    expected = expansion_jobs(root, phase, baseline, frozen)
    by_id = _unique_jobs(jobs)
    for job in expected:
        if job["id"] in by_id and canonical(by_id[job["id"]]) != canonical(job):
            raise QueueError("Existing expansion job differs from the frozen phase template")
    if all(job["id"] in by_id for job in expected):
        return {"status": "already_queued", "reason": "all 15 task IDs already exist unchanged",
                "freeze_audit_sha256": frozen["sha256"]}, []
    evidence_hashes = {str(frozen["path"]): frozen["sha256"]}
    for source in (Path(__file__), Path(__file__).with_name("pilot_gate.py"),
                   Path(__file__).with_name("queue_append.py"), Path(reporter.__file__)):
        evidence_hashes[str(source.resolve())] = file_sha256(source)
    candidate_sets = []
    for job in frozen["jobs"]:
        task, identifier = _task(job), job["id"]
        record = state.get("jobs", {}).get(identifier, {})
        if record.get("status") in ("failed", "stopped", "not_started", "unverified_or_unreaped"):
            return {"status": "rejected", "reason": f"pilot {identifier} is terminal without completion"}, []
        if record.get("status") != "complete":
            return {"status": "waiting", "reason": f"pilot {identifier} has not completed in controller"}, []
        if canonical(record.get("job")) != canonical(job):
            raise QueueError("Controller pilot record differs from its frozen/queued job")
        path = Path(job["output"])/"result.json"
        base_job = next(j for j in baseline if _task(j) == task)
        base_path = Path(base_job["output"])/"result.json"
        result, base = read_json(path), read_json(base_path)
        for value in (result, base):
            if (value.get("status") != "complete" or value.get("is_smoke") is not False
                    or (value.get("model"), value.get("dataset")) != task
                    or value.get("task") != reporter.TASKS[task]):
                raise QueueError("Pilot or phase6 reference is not a full complete matching task")
        if (not isinstance(result.get("rows"), int) or isinstance(result["rows"], bool) or result["rows"] <= 0
                or result["rows"] != base.get("rows") or result.get("support_count") != base.get("support_count")):
            raise QueueError("Pilot uses truncated rows or different support count")
        for value in (result, base):
            if (not isinstance(value.get("support_count"), int) or isinstance(value["support_count"], bool)
                    or value["support_count"] <= 0):
                raise QueueError("Invalid support count")
            for candidate in value["candidates"].values():
                for metric in reporter.METRICS[reporter.TASKS[task]]:
                    number = candidate["metrics"][metric]
                    if (not isinstance(number, (int,float)) or isinstance(number, bool)
                            or not math.isfinite(number) or not 0 <= number <= 1):
                        raise QueueError("Principal metrics must be finite non-boolean proportions")
        for value, value_path, reference_job in ((result, path, job), (base, base_path, base_job)):
            provenance = _result_provenance(value, value_path, task[0])
            hashes = provenance.get("source_sha256", {})
            for source in (reference_job["command_marker"], _argument(reference_job["argv"], "--candidate-module")):
                if hashes.get(source) != reference_job["source_sha256"].get(source):
                    raise QueueError("Result provenance differs from its frozen worker/candidate source")
            verify_sources({"source_sha256": hashes}, required=True)
            evidence_hashes.update(hashes)
            evidence_hashes[str(value_path)] = file_sha256(value_path)
            if task[0] == "alpharad":
                audit_path = value_path.parent/"data_audit.json"
                evidence_hashes[str(audit_path)] = file_sha256(audit_path)
        for control in CONTROLS:
            for metric in reporter.METRICS[reporter.TASKS[task]]:
                if not math.isclose(result["candidates"][control]["metrics"][metric],
                                    base["candidates"][control]["metrics"][metric], rel_tol=0., abs_tol=1e-9):
                    raise QueueError("Historical control metric differs from phase6 reference")
        candidate_sets.append(set(result["candidates"]))
    if any(names != candidate_sets[0] for names in candidate_sets[1:]):
        raise QueueError("Five pilots did not evaluate the same complete candidate list")
    phase_paths = sorted(str(p) for p in (root/"runs"/phase).glob("*/result.json"))
    if phase_paths != sorted(str(Path(j["output"])/"result.json") for j in frozen["jobs"]):
        raise QueueError("Pilot phase contains extra results outside the fixed five tasks")
    reported = [p for p in summary.get("phases", []) if p.get("phase") == phase]
    if len(reported) != 1:
        return {"status": "waiting", "reason": "verified summary has not reported the complete pilot phase"}, []
    current, _ = reporter.summarize_phase(root/"runs"/phase)
    if canonical(current) != canonical(reported[0]):
        return {"status": "waiting", "reason": "summary is stale for the current pilot results"}, []
    # Re-verify all native controls in the phase6 reference through the same reporter.
    baseline_report, _ = reporter.summarize_phase(root/"runs/phase6")
    if baseline_report["complete_tasks"] != 15 or baseline_report["failed_or_interrupted"]:
        raise QueueError("Phase6 reference must contain all 15 verified complete tasks")
    if current.get("shared_fitter_sha256") != frozen["fitter_hash"]:
        raise QueueError("Reported candidate source differs from the phase freeze")
    candidates, passing = candidate_evidence(current)
    for path, digest in evidence_hashes.items():
        if file_sha256(path) != digest:
            raise QueueError("Pilot evidence changed during verification")
    decision = {"status": "eligible" if passing else "rejected", "candidate_evidence": candidates,
        "passing_shared_candidates": passing, "shared_fitter_sha256": frozen["fitter_hash"],
        "freeze_audit_sha256": frozen["sha256"], "evidence_files_sha256": evidence_hashes,
        "report_phase_sha256": hashlib.sha256(canonical(current)).hexdigest(), "phase_result_paths": phase_paths,
        "expansion_policy": "all candidates; ten remaining tasks/nine pairs in phase6 task order; frozen phase CLI",
        "reason": "necessary pilot gate satisfied" if passing else "no non-control common candidate passes the pilot gate"}
    return decision, expected if passing else []


def run_once(root=ROOT, phases=ALLOWED_PHASES, deadline=AUTHORIZED_DEADLINE, *,
             commit=False, now=time.time, controller_live=_controller_is_live,
             reporter=None, persist=False):
    root = Path(root).resolve()
    if not isinstance(phases, (list, tuple)) or not phases or len(set(phases)) != len(phases) or not set(phases).issubset(ALLOWED_PHASES):
        raise QueueError("Only unique phases 12, 14, 15, 16, 17, 17b, 18 are allowed")
    if not isinstance(commit, bool):
        raise QueueError("commit must be an explicit boolean")
    guard = Guard(root, deadline, now=now, controller_live=controller_live)
    output = {"updated_at": now_iso(), "status": "running", "commit": commit,
        "interpretation": EXPLORATORY, "deadline_epoch": deadline,
        "configured_phases": list(phases), "phases": {}}
    try:
        guard()
        summary = read_json(root/"summary.json")
        if summary.get("expected_tasks") != 15 or summary.get("expected_pairs") != 14:
            raise QueueError("Expected the full 15-task/14-pair campaign summary")
        output["summary_sha256"] = file_sha256(root/"summary.json")
        reporter = load_reporter(root) if reporter is None else reporter
        jobs = queue_jobs(read_json(root/"queue_v2.json"))
        _unique_jobs(jobs)
        baseline = phase6_universe(root, jobs, reporter, deadline)
        for phase in phases:
            state = guard()
            try:
                decision, new_jobs = inspect_phase(root, phase, jobs, baseline, summary, reporter, state, deadline)
                if new_jobs:
                    snapshot = {j["id"]: j for j in jobs if j in baseline or j["id"] in {_job_id(phase,t) for t in PILOTS}}
                    def evidence_guard():
                        live = guard()
                        queued = _unique_jobs(queue_jobs(read_json(root/"queue_v2.json")))
                        for identifier, job in snapshot.items():
                            if canonical(queued.get(identifier)) != canonical(job):
                                raise QueueError("Frozen pilot/phase6 queue template changed before commit")
                            verify_sources(job, required=True)
                        for task in PILOTS:
                            identifier = _job_id(phase, task)
                            record = live.get("jobs", {}).get(identifier, {})
                            if record.get("status") != "complete" or canonical(record.get("job")) != canonical(snapshot[identifier]):
                                raise QueueError("Pilot completion changed before queue commit")
                        for path, digest in decision["evidence_files_sha256"].items():
                            if file_sha256(path) != digest:
                                raise QueueError("Pilot/source/audit evidence changed before queue commit")
                        if sorted(str(p) for p in (root/"runs"/phase).glob("*/result.json")) != decision["phase_result_paths"]:
                            raise QueueError("Pilot result set changed before queue commit")
                        latest = [p for p in read_json(root/"summary.json").get("phases", []) if p.get("phase") == phase]
                        if len(latest) != 1 or hashlib.sha256(canonical(latest[0])).hexdigest() != decision["report_phase_sha256"]:
                            raise QueueError("Verified pilot summary changed before queue commit")
                    decision["append_result"] = append_jobs(root/"queue_v2.json", new_jobs,
                        guard=evidence_guard, dry_run=not commit)
                    decision["status"] = "expanded" if commit else "eligible_dry_run"
                    jobs = queue_jobs(read_json(root/"queue_v2.json"))
                output["phases"][phase] = decision
            except (OSError, ValueError, KeyError, TypeError) as error:
                output["phases"][phase] = {"status": "error", "reason": str(error)}
        if any(p["status"] == "error" for p in output["phases"].values()):
            output["status"] = "error"
        elif all(p["status"] != "waiting" for p in output["phases"].values()):
            output["status"] = "finished"
        guard()
    except GateStopped as error:
        output.update(status="stopped", reason=str(error))
    except (OSError, ValueError, KeyError, TypeError) as error:
        output.update(status="error", reason=str(error))
    if persist:
        atomic_json(root/"pilot_gate_next_state.json", output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--phases", nargs="+", choices=ALLOWED_PHASES, default=list(ALLOWED_PHASES))
    parser.add_argument("--deadline", type=float, default=AUTHORIZED_DEADLINE)
    parser.add_argument("--commit", action="store_true", help="Explicitly append eligible full jobs under the shared queue lock")
    parser.add_argument("--no-state-file", action="store_true", help="Print evidence only; do not write this gate's separate state file")
    args = parser.parse_args(argv)
    result = run_once(args.root, args.phases, args.deadline, commit=args.commit, persist=not args.no_state_file)
    print(canonical(result).decode(), flush=True)
    return 1 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
