#!/usr/bin/env python3
"""Launch isolated Node21 adapter training on selected physical GPUs.

Only the standard library is imported here. CUDA is initialized by the trainer,
after CUDA_VISIBLE_DEVICES has been set for torchrun and its workers.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT.parent / "radzero-assets"
OUTPUTS_DIR = (ASSETS_DIR / "outputs").resolve()
FEATURES_DIR = (ASSETS_DIR / "features").resolve()
TRAINER = Path(__file__).with_name("train_node21_adapter.py")
RUN_KIND = "node21_adapter_launcher_v1"
PROTECTED = tuple((OUTPUTS_DIR / name).resolve() for name in (
    "public5_gpu0", "public5_smoke_gpu0",
    "public5_prompt_ensemble_gpu01", "public5_prompt_ensemble_smoke_gpu01",
))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class LauncherInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    raise LauncherInterrupted(signum)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--splits-dir", type=Path,
                        default=ASSETS_DIR / "splits/node21_adapter_v1")
    result.add_argument("--cache-dir", type=Path,
                        default=ASSETS_DIR / "features/node21_adapter_fp32_v1")
    result.add_argument("--gpus", nargs="+", type=int, default=[0, 1],
                        help="Physical GPU IDs; CUDA_VISIBLE_DEVICES is set before torchrun.")
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--extract-batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--limit-train", type=int, default=0)
    result.add_argument("--limit-valid", type=int, default=0)
    result.add_argument("--limit-test", type=int, default=0)
    result.add_argument("--background", action="store_true",
                        help="Detach the launcher; trainer stdout/stderr go to output/run.log.")
    return result


def validate_args(args):
    args.output = args.output.expanduser().resolve()
    args.splits_dir = args.splits_dir.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.output.parent != OUTPUTS_DIR:
        raise ValueError("Use an independent directory directly beneath " + str(OUTPUTS_DIR))
    if not args.output.relative_to(OUTPUTS_DIR).parts[0].startswith("node21_adapter_"):
        raise ValueError("Node21 adapter outputs must use a node21_adapter_* directory")
    for protected in PROTECTED:
        if (args.output == protected or protected in args.output.parents
                or args.output in protected.parents):
            raise ValueError("Existing inference results are protected: " + str(protected))
    if args.cache_dir.parent != FEATURES_DIR:
        raise ValueError("Use an independent cache directory directly beneath " + str(FEATURES_DIR))
    if not args.cache_dir.relative_to(FEATURES_DIR).parts[0].startswith("node21_adapter_"):
        raise ValueError("Node21 adapter caches must use a node21_adapter_* directory")
    cache_protected = (
        *PROTECTED, (ASSETS_DIR / "model").resolve(),
        (ASSETS_DIR / "data_images").resolve(), (ASSETS_DIR / "pretrained").resolve(),
        PROJECT_ROOT, args.splits_dir, args.output,
    )
    for protected in cache_protected:
        if (args.cache_dir == protected or protected in args.cache_dir.parents
                or args.cache_dir in protected.parents):
            raise ValueError("Cache directory overlaps protected data or results: " + str(protected))
    if args.gpus != [0, 1]:
        raise ValueError("This two-rank experiment requires --gpus 0 1")
    for key in ("epochs", "patience", "batch_size", "extract_batch_size"):
        if getattr(args, key) <= 0:
            raise ValueError("--" + key.replace("_", "-") + " must be positive")
    for key in ("workers", "limit_train", "limit_valid", "limit_test"):
        if getattr(args, key) < 0:
            raise ValueError("--" + key.replace("_", "-") + " must be nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive; weight decay must be nonnegative")
    if not TRAINER.is_file():
        raise FileNotFoundError("Trainer is missing: " + str(TRAINER))
    if not (args.splits_dir / "split_manifest.json").is_file():
        raise FileNotFoundError(
            "Prepare Node21 splits first: " + str(args.splits_dir / "split_manifest.json")
        )
    if args.output.exists() and not args.output.is_dir():
        raise ValueError("Output must be a directory")
    if args.output.exists() and any(args.output.iterdir()):
        metadata = args.output / "launcher.json"
        if not metadata.is_file() or json.loads(metadata.read_text()).get("kind") != RUN_KIND:
            raise ValueError("Refusing to reuse a nonempty directory owned by another run")


def trainer_arguments(args):
    result = []
    for key in ("output", "splits_dir", "cache_dir", "epochs", "patience",
                "batch_size", "extract_batch_size", "workers", "learning_rate",
                "weight_decay", "seed", "limit_train", "limit_valid", "limit_test"):
        result.extend(("--" + key.replace("_", "-"), str(getattr(args, key))))
    return result


def stop_group(process):
    """Terminate this launcher's torchrun process group, including child ranks."""
    if process is None:
        return
    # A torchrun parent can already have exited while descendants remain alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # Check the group itself, rather than only the torchrun parent.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def foreground(args):
    with ExitStack() as handles:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_lock = handles.enter_context((args.cache_dir / ".cache.lock").open("a"))
        try:
            fcntl.flock(cache_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This feature cache already has an active launcher") from exc
        args.output.mkdir(parents=True, exist_ok=True)
        lock = handles.enter_context((args.output / ".run.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output already has an active launcher") from exc
        metadata_path = args.output / "launcher.json"
        command = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nnodes", "1", "--nproc_per_node", str(len(args.gpus)),
            str(TRAINER), *trainer_arguments(args),
        ]
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get("status") == "complete":
                raise RuntimeError("This run is complete; use a new output directory for a new experiment")
            if previous.get("command") != command or previous.get("gpus") != args.gpus:
                raise RuntimeError("Refusing to reuse an output with different arguments or GPUs")
        status = {
            "kind": RUN_KIND, "status": "starting", "pid": os.getpid(),
            "gpus": args.gpus, "cuda_visible_devices": ",".join(map(str, args.gpus)),
            "started_at": now(), "completed_at": None, "exit_code": None,
            "command": command, "output": str(args.output),
        }
        write_json(metadata_path, status)
        (args.output / "run.pid").write_text(str(os.getpid()) + "\n")
        process = None
        try:
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=status["cuda_visible_devices"],
                               OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                               PYTHONUNBUFFERED="1")
            with (args.output / "run.log").open("a", buffering=1) as log:
                log.write("\n=== Node21 adapter launch " + status["started_at"] + " ===\n")
                log.write(json.dumps(command) + "\n")
                log.flush()
                process = subprocess.Popen(
                    command, cwd=str(PROJECT_ROOT), env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                status.update(status="running", torchrun_pid=process.pid,
                              torchrun_process_group=process.pid)
                write_json(metadata_path, status)
                print("Node21 adapter launched: " + str(args.output), flush=True)
                exit_code = process.wait()
            if exit_code != 0:
                stop_group(process)
            status.update(status="complete" if exit_code == 0 else "failed",
                          exit_code=exit_code, completed_at=now())
            write_json(metadata_path, status)
            return exit_code
        except BaseException as exc:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            stop_group(process)
            exit_code = 128 + exc.signum if isinstance(exc, LauncherInterrupted) else 1
            status.update(status="failed", exit_code=exit_code, completed_at=now(),
                          error=repr(exc), traceback=traceback.format_exc())
            write_json(metadata_path, status)
            print(status["error"], file=sys.stderr, flush=True)
            return exit_code


def background(args):
    command = [sys.executable, str(Path(__file__).resolve()), *trainer_arguments(args),
               "--gpus", *map(str, args.gpus)]
    process = subprocess.Popen(
        command, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        # A small handshake reports lock/configuration failures to the caller.
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            exit_code = process.poll()
            path = args.output / "launcher.json"
            if path.exists():
                status = json.loads(path.read_text())
                if status.get("pid") == process.pid:
                    if status.get("status") in ("running", "complete"):
                        print(json.dumps({
                            "launcher_pid": process.pid, "output": str(args.output),
                            "log": str(args.output / "run.log"),
                            "status": status["status"],
                        }, indent=2))
                        return 0
                    if status.get("status") == "failed":
                        raise RuntimeError(status.get("error", "Trainer launch failed; inspect run.log"))
            if exit_code is not None:
                raise RuntimeError(
                    "Detached launcher exited with code " + str(exit_code)
                    + "; check for an existing active/completed run or inspect launcher.json"
                )
            time.sleep(0.1)
        raise RuntimeError("Detached launcher did not confirm startup within 15 seconds")
    except BaseException:
        # Ask the child launcher to clean up its separate torchrun group.
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def main():
    args = parser().parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        validate_args(args)
        return background(args) if args.background else foreground(args)
    except LauncherInterrupted as exc:
        return 128 + exc.signum
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
