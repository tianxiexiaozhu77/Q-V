#!/usr/bin/env python3
"""Run independent RadZero visual I2CL experiments on idle physical GPUs 2–7."""

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
import sys
import time
import uuid

from asset_paths import ASSETS_DIR

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = ASSETS_DIR
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
TRAINER = Path(__file__).with_name("train_i2cl.py")
GPU_LOCKS = PROJECT.parent / "RadZero/runs/gpu_locks"
ALLOWED_GPUS = tuple(range(2, 8))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=ASSETS / "outputs" / datetime.now().strftime("public5_i2cl_%Y%m%d_%H%M%S"))
    p.add_argument(
        "--split-dir", type=Path,
        help="Default: native_public5_local_seed42 for 5 shots/seed 42; otherwise a generated split",
    )
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--gpus", nargs="+", type=int, choices=ALLOWED_GPUS, default=list(ALLOWED_GPUS))
    p.add_argument("--shots-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument(
        "--lambda-reg", type=float, default=0.01,
        help="Identity coefficient penalty strength (default: 0.01; 0 disables it)",
    )
    p.add_argument("--noise-scale", type=float, default=0.001)
    p.add_argument("--limit-test", type=int, default=0, help="0: full held-out test; positive: smoke only.")
    p.add_argument("--background", action="store_true")
    p.add_argument("--_run-token", help=argparse.SUPPRESS)
    return p


def validate(args):
    args.output = args.output.expanduser().resolve()
    outputs = (ASSETS / "outputs").resolve()
    if args.output.parent != outputs:
        raise ValueError("Use a new direct child directory of " + str(outputs))
    if args.split_dir is None:
        if args.shots_per_class == 5 and args.seed == 42:
            args.split_dir = ASSETS / "splits/native_public5_local_seed42"
        else:
            args.split_dir = ASSETS / f"splits/public5_i2cl_shot{args.shots_per_class}_seed{args.seed}_v1"
    args.split_dir = args.split_dir.expanduser().resolve()
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate GPU or dataset selections")
    if not args.gpus or not set(args.gpus).issubset(ALLOWED_GPUS):
        raise ValueError("Only physical GPUs 2–7 are authorized")
    for key in ("shots_per_class", "epochs", "grad_batch_size", "eval_batch_size"):
        if getattr(args, key) < 1:
            raise ValueError(key + " must be positive")
    for key in ("workers", "limit_test", "weight_decay", "noise_scale"):
        if getattr(args, key) < 0:
            raise ValueError(key + " must be nonnegative")
    if args.learning_rate <= 0 or args.seed < 0:
        raise ValueError("Invalid learning rate or seed")
    if not math.isfinite(args.lambda_reg) or args.lambda_reg < 0:
        raise ValueError("lambda-reg must be finite and nonnegative")
    if not TRAINER.is_file():
        raise FileNotFoundError(TRAINER)


def claim(args):
    if args._run_token:
        request = json.loads((args.output / "launch_request.json").read_text())
        if request["token"] != args._run_token:
            raise ValueError("Output belongs to a different launcher")
        return args._run_token
    # Never resume or overwrite existing experiments, including an interrupted run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    token = uuid.uuid4().hex
    config = {key: str(v) if isinstance(v, Path) else v
              for key, v in vars(args).items() if not key.startswith("_")}
    with (args.output / "launch_request.json").open("x") as f:
        json.dump({"token": token, "requested_at": now(), "config": config}, f, indent=2)
    return token


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def interrupted(signum, _frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def reserve_idle_gpu(index):
    """Hold the shared lease before inspecting the physical GPU by UUID.

    The scheduler owns this descriptor. Workers never inherit it, including
    forked data-loader workers, so an exited task cannot leave a live lease.
    """
    if index not in ALLOWED_GPUS:
        raise ValueError("Only physical GPUs 2–7 are authorized")
    GPU_LOCKS.mkdir(parents=True, exist_ok=True)
    lock = (GPU_LOCKS / f"eval_gpu_{index}.lock").open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return None, {"reason": "shared_lease_busy"}
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", str(index),
             "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip()
        gpu_uuid, name, memory = [value.strip() for value in raw.split(",")]
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        busy = [row for row in rows.splitlines()
                if row.split(",")[0].strip() == gpu_uuid]
        info = {"physical_index": index, "uuid": gpu_uuid, "name": name,
                "memory_used_before_mib": int(memory)}
        if busy or int(memory) > 512:
            info.update(reason="gpu_in_use", processes=busy)
            lock.close()
            return None, info
        return lock, info
    except BaseException:
        lock.close()
        raise


def summarize(output, state):
    """Refresh a human-readable status and paired metrics without hiding failures."""
    lines = [
        "RadZero_i2cl: " + state["status"],
        "Allowed physical GPUs: " + ", ".join(map(str, state["physical_gpus"])),
        "Identity regularization lambda: " + str(state.get("lambda_reg", 0.0)),
        "Dataset | Status | Original AUROC | I2CL AUROC | Original AUPRC | I2CL AUPRC",
    ]
    summary = {}
    for name in state["datasets"]:
        job = state["jobs"].get(name, {})
        row = {"status": job.get("status", "pending")}
        result_path = output / name / "result.json"
        if row["status"] == "complete" and result_path.is_file():
            result = json.loads(result_path.read_text())
            row.update(baseline_metrics=result["baseline_metrics"], metrics=result["metrics"],
                       delta_pp=result["delta_pp"], support_count=result["support_count"],
                       test_count=result["evaluated_count"],
                       is_smoke_test=result["is_smoke_test"])
        summary[name] = row
        metrics = []
        for key in ("macro_auroc", "macro_auprc"):
            for method in ("baseline_metrics", "metrics"):
                value = row.get(method, {}).get(key)
                metrics.append(f"{100 * value:.2f}%" if value is not None else "—")
        lines.append(" | ".join([name, row["status"], *metrics]))
    if state["is_smoke"]:
        lines.append("SMOKE ONLY: these metrics are not a full held-out evaluation.")
    write_json(output / "summary.json", {
        "status": state["status"], "lambda_reg": state.get("lambda_reg", 0.0),
        "datasets": summary,
    })
    target = output / "RESULTS.txt"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(target)


def run(args):
    from i2cl_data import load_spec, prepare_splits

    state = {"status": "preparing_splits", "started_at": now(), "pid": os.getpid(),
             "physical_gpus": args.gpus, "datasets": args.datasets,
             "is_smoke": bool(args.limit_test),
             "lambda_reg": args.lambda_reg,
             "jobs": {name: {"status": "pending"} for name in args.datasets},
             "split_dir": str(args.split_dir),
             "source_sha256": {}}
    for filename in ("run_i2cl.py", "train_i2cl.py", "i2cl_model.py", "i2cl_data.py"):
        path = Path(__file__).with_name(filename)
        state["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = args.output / "launcher.json"
    started = time.monotonic()
    active = {}
    try:
        write_json(path, state)
        split_manifest = args.split_dir / "manifest.json"
        split_dirs = {}
        if split_manifest.is_file():
            prepared = json.loads(split_manifest.read_text())
            if (prepared["seed"] != args.seed
                    or prepared["shots_per_class"] != args.shots_per_class):
                raise ValueError("Existing splits differ from the requested seed or quota")
        else:
            prepared = None
        for name in args.datasets:
            try:
                if prepared is not None:
                    if name not in prepared["datasets"]:
                        raise ValueError(f"{name} is missing from the requested split manifest")
                    dataset_split_dir = args.split_dir
                else:
                    # A missing dataset must not prevent independent datasets
                    # from being prepared, trained and evaluated.
                    dataset_split_dir = args.split_dir / "_datasets" / name
                    prepare_splits(dataset_split_dir, shots_per_class=args.shots_per_class,
                                   seed=args.seed, datasets=[name])
                for split in ("support", "test"):
                    load_spec(dataset_split_dir, name, split)
                split_dirs[name] = dataset_split_dir
                state["jobs"][name]["split_dir"] = str(dataset_split_dir)
            except Exception as exc:
                state["jobs"][name].update(status="preparation_failed", error=repr(exc),
                                           finished_at=now())
                print(f"{now()} {name}: split preparation failed: {exc}", flush=True)
            write_json(path, state)
        # Each task trains and evaluates one dataset independently.
        order = ("ChestXray14", "OpenI", "RSNA", "SIIM", "ChestXDet10")
        pending = [name for name in order if name in split_dirs]
        state["status"] = "running"
        last_wait_report = 0.0
        while pending or active:
            for gpu in args.gpus:
                if gpu in active or not pending:
                    continue
                try:
                    lease, gpu_info = reserve_idle_gpu(gpu)
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    state.setdefault("gpu_checks", {})[str(gpu)] = {
                        "checked_at": now(), "reason": "query_failed", "error": str(exc)}
                    continue
                state.setdefault("gpu_checks", {})[str(gpu)] = {
                    "checked_at": now(), **gpu_info}
                if lease is None:
                    continue
                name = pending.pop(0)
                directory = args.output / name
                state["jobs"][name] = {
                    "status": "launching", "gpu": gpu, "gpu_info": gpu_info,
                    "started_at": now(), "log": str(directory / "worker.log"),
                    "split_dir": str(split_dirs[name])}
                log = None
                try:
                    directory.mkdir()
                    log = (directory / "worker.log").open("w")
                    command = [sys.executable, "-u", str(TRAINER), "--dataset", name,
                               "--split-dir", str(split_dirs[name]), "--output", str(directory),
                               "--device", "cuda"]
                    for key in ("seed", "epochs", "grad_batch_size", "eval_batch_size",
                                "workers", "learning_rate", "weight_decay", "lambda_reg",
                                "noise_scale", "limit_test"):
                        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu_info["uuid"],
                               RADZERO_PHYSICAL_GPU=str(gpu), TOKENIZERS_PARALLELISM="false",
                               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
                    process = subprocess.Popen(
                        command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, close_fds=True)
                    active[gpu] = (process, name, lease)
                    state["jobs"][name].update(status="running", pid=process.pid, command=command)
                    print(f"{now()} {name}: physical GPU {gpu}, PID {process.pid}", flush=True)
                except Exception as exc:
                    lease.close()
                    state["jobs"][name].update(status="failed", error=repr(exc), finished_at=now())
                    print(f"{now()} {name}: failed to launch: {exc}", flush=True)
                finally:
                    if log is not None:
                        log.close()
                write_json(path, state)
            for gpu, (process, name, lease) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                result_path = args.output / name / "result.json"
                try:
                    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
                    complete = code == 0 and result.get("status") == "complete"
                except (OSError, ValueError) as exc:
                    complete = False
                    state["jobs"][name]["result_error"] = repr(exc)
                status = "complete" if complete else "failed"
                state["jobs"][name].update(status=status, exit_code=code, finished_at=now())
                lease.close()
                del active[gpu]
                print(f"{now()} {name}: {status} (exit {code})", flush=True)
            if pending and not active and time.monotonic() - last_wait_report >= 60:
                print(f"{now()} Waiting for an idle authorized GPU; pending: {', '.join(pending)}",
                      flush=True)
                last_wait_report = time.monotonic()
            write_json(path, state)
            summarize(args.output, state)
            if pending or active:
                time.sleep(5)
        complete = all(job["status"] == "complete" for job in state["jobs"].values())
        state.update(status="complete" if complete else "partial_failure",
                     finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(path, state)
        summarize(args.output, state)
        return complete
    except BaseException as exc:
        for process, name, lease in active.values():
            try:
                stop(process)
                state["jobs"][name].update(status="terminated", exit_code=process.returncode)
            finally:
                lease.close()
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=str(exc), finished_at=now(), elapsed_seconds=time.monotonic() - started)
        write_json(path, state)
        summarize(args.output, state)
        raise


def main():
    args = parser().parse_args()
    validate(args)
    token = claim(args)
    if args.background:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[arg for arg in sys.argv[1:] if arg != "--background"], "--_run-token", token]
        with (args.output / "run.log").open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
        print(json.dumps({"pid": process.pid, "output": str(args.output),
                          "log": str(args.output / "run.log")}, indent=2))
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    if not run(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
