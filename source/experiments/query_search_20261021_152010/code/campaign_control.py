#!/usr/bin/env python3
"""Run an appendable JSON job queue on idle GPUs 3–7 until a strict deadline.

Queue example (write updates using rename/replace):
{"jobs": [{"id": "trial_OpenI", "dataset": "OpenI",
           "argv": ["bash", "local_setup/python_local.sh", "worker.py",
                    "--output", "/absolute/new/output"],
           "output": "/absolute/new/output",
           "command_marker": "worker.py", "create_output": false}]}

The controller owns state and logs, never edits the queue, and never retries a
completed/failed job ID. Append a new ID and output for another experiment.
An absent/empty queue waits for additions until deadline. Touch STOP to stop.
"""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

import sys
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "AlphaRAD/local_setup"))
from run_pretrained import reserve_gpu as reserve_idle_gpu

AUTHORIZED_GPUS = (3, 4, 5, 6, 7)
HARD_DEADLINE = datetime.fromisoformat("2026-10-22T10:00:00+08:00").timestamp()
TOKEN_KEY = "UNIFIED_QUERY_CAMPAIGN_JOB_TOKEN"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid):
    """Read Linux process birth time, group/session and actual command."""
    try:
        proc = Path("/proc") / str(pid)
        raw = (proc / "stat").read_text()
        # comm may contain spaces/parentheses; fields after its final ')' start at 3.
        fields = raw[raw.rindex(")") + 2:].split()
        command = (proc / "cmdline").read_bytes().split(b"\0")
        command = [part.decode(errors="replace") for part in command if part]
        environment = (proc / "environ").read_bytes().split(b"\0")
        prefix = (TOKEN_KEY + "=").encode()
        token = next((part[len(prefix):].decode() for part in environment
                      if part.startswith(prefix)), None)
        return {"pid": int(pid), "state": fields[0], "pgid": int(fields[2]),
                "session": int(fields[3]), "start_ticks": int(fields[19]),
                "uid": proc.stat().st_uid, "argv": command, "token": token}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def owned_group_members(record):
    """Require job token, UID, session, birth time and command evidence.

    The process-group leader may exec the Python wrapper; the stable marker must
    remain in its actual argv. Token-carrying children remain identifiable after
    the leader exits. Unrelated/reused PIDs are never accepted.
    """
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        # Cheap stat filter first: do not open every other user's environment
        # or command line while checking our own small process group.
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()
            if int(fields[2]) != record["pid"]:
                continue
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
        identity = process_identity(int(entry.name))
        if identity is None or identity["pgid"] != record["pid"]:
            continue
        if identity["state"] == "Z":
            continue
        valid = (identity["session"] == record["pid"]
                 and identity["uid"] == os.getuid()
                 and identity["token"] == record["token"]
                 and identity["start_ticks"] >= record["start_ticks"])
        if identity["pid"] == record["pid"]:
            valid = (valid and identity["start_ticks"] == record["start_ticks"]
                     and record["command_marker"] in identity["argv"])
        if not valid:
            # Never send a group signal when any live member cannot be verified.
            return None
        members.append(identity)
    return members


def signal_owned_group(record, signum):
    members = owned_group_members(record)
    if members is None:
        return False
    if not members:
        return True
    try:
        os.killpg(record["pid"], signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        return False
    return True


def validate_job(raw):
    if not isinstance(raw, dict):
        raise ValueError("Each job must be an object")
    job = dict(raw)
    identifier = job.get("id")
    if (not isinstance(identifier, str) or not identifier
            or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                   for char in identifier)):
        raise ValueError("Job id must contain only letters, digits, _, - or .")
    argv = job.get("argv")
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)):
        raise ValueError("argv must be a nonempty list of strings")
    if not job.get("dataset") or not job.get("output"):
        raise ValueError("dataset and output are required")
    job["output"] = str(Path(job["output"]).expanduser().resolve())
    job["cwd"] = str(Path(job.get("cwd", PROJECT)).expanduser().resolve())
    job["command_marker"] = job.get("command_marker", argv[-1])
    if not job["command_marker"] or job["command_marker"] not in argv:
        raise ValueError("command_marker must be a nonempty exact argv element")
    if not isinstance(job.get("create_output", False), bool):
        raise ValueError("create_output must be boolean")
    return job


class Controller:
    def __init__(self, args, reserve=reserve_idle_gpu):
        if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
                or not set(args.gpus).issubset(AUTHORIZED_GPUS)):
            raise ValueError("Only unique physical GPUs 3–7 are authorized")
        self.args = args
        self.reserve = reserve
        self.active = {}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.stop_reason = None
        self.state = {"status": "running", "started_at": now(), "pid": os.getpid(),
                      "deadline": datetime.fromtimestamp(args.deadline, timezone.utc).isoformat(),
                      "gpus": args.gpus, "jobs": {}, "events": []}
        self.state_path = args.state_dir / "state.json"
        self.claimed_outputs = set()

    def save(self):
        with self.lock:
            snapshot = copy.deepcopy(self.state)
        # A stalled NFS write must not hold the watchdog's ownership lock.
        atomic_json(self.state_path, snapshot)

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
                self.state["status"] = "stopping"
                self.state["stop_reason"] = reason

    def watchdog(self):
        # Runs independently of nvidia-smi/reservation calls.
        while not self.stopping.is_set():
            if time.time() >= self.args.deadline - self.args.grace_seconds:
                self.request_stop("deadline")
                break
            if self.args.stop_file.exists():
                self.request_stop("stop_file")
                break
            self.stopping.wait(min(0.2, max(0.001, self.args.deadline - self.args.grace_seconds - time.time())))
        with self.lock:
            records = [dict(value["record"]) for value in self.active.values()]
        for record in records:
            if not signal_owned_group(record, signal.SIGTERM):
                self.event(f"Refused TERM for unverifiable process group {record['pid']}")
        end = time.monotonic() + self.args.grace_seconds
        while time.monotonic() < end:
            if all(owned_group_members(record) == [] for record in records):
                break
            time.sleep(0.1)
        for record in records:
            if not signal_owned_group(record, signal.SIGKILL):
                self.event(f"Refused KILL for unverifiable process group {record['pid']}")

    def read_queue(self):
        if not self.args.queue.exists():
            return []
        raw = json.loads(self.args.queue.read_text())
        jobs = raw if isinstance(raw, list) else raw["jobs"]
        if not isinstance(jobs, list):
            raise ValueError("queue jobs must be a list")
        return jobs

    def accept_queue(self):
        try:
            raw_jobs = self.read_queue()
        except Exception as exc:
            self.event(f"Queue temporarily unreadable: {exc!r}")
            return
        for raw in raw_jobs:
            try:
                job = validate_job(raw)
                identifier = job["id"]
                if identifier in self.state["jobs"]:
                    if self.state["jobs"][identifier]["job"] != job:
                        self.event(f"Ignoring modified existing job {identifier}; use a new id")
                    continue
                if job["output"] in self.claimed_outputs:
                    raise ValueError("Output already claimed by another job")
                self.claimed_outputs.add(job["output"])
                self.state["jobs"][identifier] = {
                    "status": "pending", "queued_at": now(), "job": job}
            except Exception as exc:
                self.event(f"Rejected queue entry: {exc!r}")

    def launch(self, identifier, gpu, lease, gpu_info):
        record = self.state["jobs"][identifier]
        job = record["job"]
        output = Path(job["output"])
        log_path = self.args.state_dir / "logs" / (identifier + ".log")
        log = None
        try:
            for source, expected in job.get("source_sha256", {}).items():
                if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Frozen source changed: {source}")
            if output.exists():
                raise FileExistsError(f"Refusing existing worker output {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            if job.get("create_output", False):
                output.mkdir()
            log = log_path.open("xb", buffering=0)
            environment = os.environ.copy()
            token = uuid.uuid4().hex
            environment.update(CUDA_VISIBLE_DEVICES=gpu_info["uuid"],
                               CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS="4",
                               OPENBLAS_NUM_THREADS="4",
                               MKL_NUM_THREADS="4")
            environment[TOKEN_KEY] = token
            # Serialize spawn with the watchdog: every spawned process is
            # registered before the watchdog can take its termination snapshot.
            with self.lock:
                if self.stopping.is_set() or time.time() >= self.args.deadline:
                    self.request_stop("deadline" if time.time() >= self.args.deadline else "stopped")
                    log.close()
                    lease.close()
                    return
                process = subprocess.Popen(
                    job["argv"], cwd=job["cwd"], env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                identity = process_identity(process.pid)
                record.update(status="running", started_at=now(), pid=process.pid,
                              physical_gpu=gpu, gpu_uuid=gpu_info["uuid"],
                              gpu_info=dict(gpu_info),
                              token=token, command_marker=job["command_marker"],
                              start_ticks=identity["start_ticks"] if identity else -1,
                              launch_argv_sha256=hashlib.sha256(
                                  json.dumps(job["argv"]).encode()).hexdigest(),
                              log=str(log_path))
                self.active[identifier] = {"process": process, "record": record,
                                           "lease": lease, "log": log}
            self.event(f"Started {identifier} on GPU {gpu}, PID {process.pid}")
            self.save()
        except Exception as exc:
            if identifier in self.active:
                # A persistence failure after spawn must stop the owned worker,
                # rather than accidentally releasing its lease while it runs.
                self.request_stop("post_launch_failure")
                raise
            if log is not None:
                log.close()
            lease.close()
            record.update(status="failed", finished_at=now(), error=repr(exc))
            self.event(f"Launch failed {identifier}: {exc!r}")
            self.save()

    def reap(self):
        for identifier, active in list(self.active.items()):
            code = active["process"].poll()
            if code is None:
                continue
            record = active["record"]
            members = owned_group_members(record)
            if members:
                # The root exited while its own workers remain. Keep the GPU
                # lease until they exit; deadline/stop watchdog owns cleanup.
                record["leader_returncode"] = code
                continue
            if members is None:
                record["ownership_warning"] = "Process group identity cannot be verified"
                # Retain the lease rather than let another job collide with
                # live members whose ownership could not be established.
                self.request_stop("unverifiable_process_group")
                continue
            result_ok = code == 0
            if result_ok and record["job"].get("require_result", False):
                try:
                    result = json.loads((Path(record["job"]["output"]) / "result.json").read_text())
                    result_ok = result.get("status") == "complete"
                    for key in ("dataset", "model", "is_smoke"):
                        if key in record["job"]:
                            result_ok &= result.get(key) == record["job"][key]
                    if not result_ok:
                        record["error"] = "Worker exited 0 without complete matching result"
                except Exception as exc:
                    result_ok = False
                    record["error"] = f"Worker result unavailable: {exc!r}"
            with self.lock:
                record.update(status=("stopped" if self.stopping.is_set()
                                      else "complete" if result_ok else "failed"),
                              returncode=code, finished_at=now())
                active["lease"].close()
                active["log"].close()
                del self.active[identifier]
            self.event(f"Finished {identifier}: {record['status']} ({code})")

    def run(self):
        self.args.state_dir.mkdir(parents=True, exist_ok=True)
        (self.args.state_dir / "logs").mkdir(exist_ok=True)
        with (self.args.state_dir / "controller.lock").open("a+") as controller_lock:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.state_path.exists():
                raise FileExistsError("Use a fresh state directory; refusing to reuse old state")
            self.save()
            watchdog = threading.Thread(target=self.watchdog, daemon=True)
            watchdog.start()
            try:
                while not self.stopping.is_set():
                    self.accept_queue()
                    self.reap()
                    for identifier, record in list(self.state["jobs"].items()):
                        if record["status"] != "pending" or self.stopping.is_set():
                            continue
                        busy = {item["record"]["physical_gpu"] for item in self.active.values()}
                        for gpu in self.args.gpus:
                            if self.stopping.is_set():
                                break
                            if gpu in busy:
                                continue
                            try:
                                lease, info = self.reserve(gpu)
                            except Exception as exc:
                                self.event(f"GPU {gpu} check failed: {exc!r}")
                                continue
                            if lease is None:
                                continue
                            self.launch(identifier, gpu, lease, info)
                            break
                    self.save()
                    self.stopping.wait(self.args.poll_seconds)
            except BaseException:
                self.request_stop("controller_exception")
                raise
            finally:
                self.request_stop("controller_exit")
                watchdog.join()
                # Reap briefly after KILL. Never block indefinitely on a worker.
                until = time.monotonic() + 2
                while self.active and time.monotonic() < until:
                    self.reap()
                    time.sleep(0.05)
                for identifier, active in self.active.items():
                    active["record"].update(
                        status="unverified_or_unreaped", finished_at=now())
                    active["lease"].close()
                    active["log"].close()
                for record in self.state["jobs"].values():
                    if record["status"] == "pending":
                        record.update(status="not_started", finished_at=now())
                self.state.update(status="stopped", finished_at=now())
                self.save()
        return self.state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(AUTHORIZED_GPUS))
    parser.add_argument("--deadline", default="2026-10-22T10:00:00+08:00")
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--grace-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or not set(args.gpus).issubset(AUTHORIZED_GPUS)):
        parser.error("Only unique physical GPUs 3–7 are authorized")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("Deadline must include a timezone")
    args.deadline = deadline.timestamp()
    if args.deadline > HARD_DEADLINE:
        parser.error("Deadline cannot exceed the authorized 2026-10-22 10:00 Asia/Shanghai")
    if args.poll_seconds <= 0 or not 0 <= args.grace_seconds <= 30:
        parser.error("poll-seconds must be positive; grace-seconds must be in [0, 30]")
    args.queue = args.queue.expanduser().resolve()
    args.state_dir = args.state_dir.expanduser().resolve()
    args.stop_file = (args.stop_file or args.state_dir / "STOP").expanduser().resolve()
    return args


def main():
    args = parse_args()
    controller = Controller(args)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
