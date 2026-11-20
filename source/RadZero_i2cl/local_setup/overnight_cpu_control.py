#!/usr/bin/env python3
"""Wait for verified frozen-feature caches, then run CPU-only OOF methods."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from overnight_control import (
    HARD_DEADLINE, PROJECT, TOKEN_KEY, atomic_json, now, owned_group_members,
    process_identity, signal_owned_group,
)

DATASETS = ("OpenI", "ChestXray14", "ChestXDet10", "SIIM", "RSNA")
METHODS = {
    "cache_oof": "overnight_cache_methods.py",
    "linear_oof": "overnight_linear_residual.py",
}
TERMINAL = {"complete", "failed", "skipped", "stopped", "not_started"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CPUController:
    def __init__(self, args):
        self.args = args
        if not 1 <= args.max_parallel <= 2:
            raise ValueError("At most two CPU workers are authorized")
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.active = {}
        self.stop_reason = None
        self.state_path = args.state_dir / "state.json"
        jobs = {}
        for dataset in args.datasets:
            for method, worker in METHODS.items():
                identifier = f"{method}_{dataset}"
                jobs[identifier] = {
                    "id": identifier, "dataset": dataset, "method": method,
                    "worker": str(PROJECT / "local_setup" / worker),
                    "worker_sha256": sha256(PROJECT / "local_setup" / worker),
                    "cache": str(args.raw_root / "features" / dataset),
                    "output": str(args.raw_root / method / dataset),
                    "status": "waiting_for_cache",
                }
        self.state = {
            "status": "running", "started_at": now(), "pid": os.getpid(),
            "deadline": datetime.fromtimestamp(args.deadline, timezone.utc).isoformat(),
            "cuda_visible_devices": "", "threads_per_worker": 4,
            "max_parallel": args.max_parallel, "jobs": jobs, "events": [],
            "stop_files": [str(path) for path in args.stop_files],
        }

    def save(self):
        with self.lock:
            atomic_json(self.state_path, self.state)

    def event(self, message):
        with self.lock:
            self.state["events"].append({"at": now(), "message": str(message)})
            self.state["events"] = self.state["events"][-200:]
        print(now(), message, flush=True)

    def request_stop(self, reason):
        with self.lock:
            if not self.stopping.is_set():
                self.stop_reason = reason
                self.stopping.set()
                self.state.update(status="stopping", stop_reason=reason)

    def watchdog(self):
        while not self.stopping.is_set():
            if time.time() >= self.args.deadline:
                self.request_stop("deadline")
                break
            if any(path.exists() for path in self.args.stop_files):
                self.request_stop("stop_file")
                break
            self.stopping.wait(min(.2, max(.001, self.args.deadline - time.time())))
        with self.lock:
            records = [dict(value["record"]) for value in self.active.values()]
        for record in records:
            if not signal_owned_group(record, signal.SIGTERM):
                self.event(f"Refused TERM for unverifiable group {record['pid']}")
        until = time.monotonic() + self.args.grace_seconds
        while time.monotonic() < until:
            if all(owned_group_members(record) == [] for record in records):
                break
            time.sleep(.1)
        for record in records:
            if not signal_owned_group(record, signal.SIGKILL):
                self.event(f"Refused KILL for unverifiable group {record['pid']}")

    def refresh_dependencies(self):
        try:
            gpu_state = json.loads(self.args.gpu_state.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            gpu_state = {}
        for job in self.state["jobs"].values():
            if job["status"] != "waiting_for_cache":
                continue
            cache = Path(job["cache"])
            try:
                manifest = json.loads((cache / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                manifest = {}
            if manifest.get("status") == "complete":
                job.update(status="pending", cache_ready_at=now(),
                           cache_manifest_sha256=sha256(cache / "manifest.json"))
                continue
            try:
                progress = json.loads((cache / "progress.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {}
            producer = gpu_state.get("jobs", {}).get("features_" + job["dataset"], {})
            failure = None
            if manifest.get("status") in ("failed", "deadline", "stopped"):
                failure = "cache_manifest_" + manifest["status"]
            elif progress.get("status") in ("failed", "deadline", "stopped"):
                failure = "cache_progress_" + progress["status"]
            elif producer.get("status") in (
                    "failed", "stopped", "not_started", "unverified_or_unreaped"):
                failure = "gpu_producer_" + producer["status"]
            elif producer.get("status") == "complete":
                failure = "gpu_producer_exited_without_complete_cache"
            if failure:
                job.update(status="skipped", reason=failure, finished_at=now())
                self.event(f"Skipping {job['id']}: {failure}")

    def command_for_job(self, job):
        argv = [
            "bash", str(PROJECT / "local_setup/python_local.sh"), job["worker"],
            "--cache", job["cache"], "--output", job["output"],
            "--dataset", job["dataset"], "--threads", "4",
            "--deadline", str(self.args.deadline),
        ]
        if job["method"] == "cache_oof":
            argv.extend(["--top-k", "9"])
        return argv

    def launch(self, job):
        identifier, log = job["id"], None
        try:
            if sha256(job["worker"]) != job["worker_sha256"]:
                raise ValueError("Worker code changed since controller launch")
            if Path(job["output"]).exists():
                raise FileExistsError(f"Refusing existing output {job['output']}")
            if sha256(Path(job["cache"]) / "manifest.json") != job["cache_manifest_sha256"]:
                raise ValueError("Cache manifest changed after becoming complete")
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            argv = self.command_for_job(job)
            log_path = self.args.state_dir / "logs" / (identifier + ".log")
            log = log_path.open("xb", buffering=0)
            environment = os.environ.copy()
            token = uuid.uuid4().hex
            environment.update(
                CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                NUMEXPR_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
            environment[TOKEN_KEY] = token
            with self.lock:
                if self.stopping.is_set() or time.time() >= self.args.deadline:
                    self.request_stop("deadline" if time.time() >= self.args.deadline else "stopped")
                    log.close()
                    return
                process = subprocess.Popen(
                    argv, cwd=PROJECT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                identity = process_identity(process.pid)
                job.update(status="running", started_at=now(), pid=process.pid,
                           token=token, start_ticks=identity["start_ticks"] if identity else -1,
                           command_marker=job["worker"], argv=argv, log=str(log_path))
                self.active[identifier] = {"record": job, "process": process, "log": log}
            self.event(f"Started CPU {identifier}, PID {process.pid}")
            self.save()
        except Exception as exc:
            if identifier in self.active:
                self.request_stop("post_launch_failure")
                raise
            if log is not None:
                log.close()
            job.update(status="failed", error=repr(exc), finished_at=now())
            self.event(f"Launch failed {identifier}: {exc!r}")

    def reap(self):
        for identifier, active in list(self.active.items()):
            code = active["process"].poll()
            if code is None:
                continue
            record = active["record"]
            members = owned_group_members(record)
            if members:
                record["leader_returncode"] = code
                continue
            if members is None:
                record["ownership_warning"] = "Process-group identity cannot be verified"
            status_path = Path(record["output"]) / "status.json"
            try:
                worker_state = json.loads(status_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                worker_state = {}
            if self.stopping.is_set():
                status = "stopped"
            else:
                status = ("complete" if code == 0 and worker_state.get("status") == "complete"
                          else "failed")
            with self.lock:
                record.update(status=status, returncode=code, finished_at=now(),
                              worker_status=worker_state.get("status"))
                if status == "failed" and code == 0:
                    record["error"] = "Worker exited zero without status.json complete"
                active["log"].close()
                del self.active[identifier]
            self.event(f"Finished CPU {identifier}: {status} ({code})")

    def run(self):
        self.args.state_dir.mkdir(parents=True, exist_ok=True)
        (self.args.state_dir / "logs").mkdir(exist_ok=True)
        with (self.args.state_dir / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.state_path.exists():
                raise FileExistsError("Refusing to reuse an existing CPU controller state")
            self.save()
            watchdog = threading.Thread(target=self.watchdog, daemon=True)
            watchdog.start()
            try:
                while not self.stopping.is_set():
                    self.reap()
                    self.refresh_dependencies()
                    for job in self.state["jobs"].values():
                        if self.stopping.is_set() or len(self.active) >= self.args.max_parallel:
                            break
                        if job["status"] == "pending":
                            self.launch(job)
                    self.save()
                    if all(job["status"] in TERMINAL for job in self.state["jobs"].values()):
                        self.request_stop("all_jobs_terminal")
                        break
                    self.stopping.wait(self.args.poll_seconds)
            except BaseException:
                self.request_stop("controller_exception")
                raise
            finally:
                self.request_stop("controller_exit")
                watchdog.join()
                until = time.monotonic() + 2
                while self.active and time.monotonic() < until:
                    self.reap()
                    time.sleep(.05)
                for active in self.active.values():
                    active["record"].update(status="unverified_or_unreaped", finished_at=now())
                    active["log"].close()
                for job in self.state["jobs"].values():
                    if job["status"] in ("pending", "waiting_for_cache"):
                        job.update(status="not_started", finished_at=now())
                if self.stop_reason == "all_jobs_terminal":
                    final = ("complete" if all(job["status"] == "complete"
                                               for job in self.state["jobs"].values())
                             else "finished_with_failures")
                else:
                    final = "stopped"
                self.state.update(status=final, finished_at=now())
                self.save()
        return self.state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--gpu-state", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, action="append", default=[])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--max-parallel", type=int, choices=(1, 2), default=2)
    parser.add_argument("--deadline", default="2026-10-21T10:00:00+08:00")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--grace-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    instant = datetime.fromisoformat(args.deadline)
    if instant.tzinfo is None or instant.timestamp() > HARD_DEADLINE:
        parser.error("Deadline must include timezone and be no later than 2026-10-21 10:00 +08")
    args.deadline = instant.timestamp()
    if (not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0
            or not math.isfinite(args.grace_seconds) or not 0 <= args.grace_seconds <= 30):
        parser.error("Invalid poll/grace duration")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("Duplicate datasets")
    for name in ("raw_root", "state_dir", "gpu_state"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.stop_files = [path.expanduser().resolve() for path in args.stop_file]
    args.stop_files += [args.state_dir / "STOP", args.gpu_state.parent / "STOP"]
    return args


def main():
    controller = CPUController(parse_args())
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
