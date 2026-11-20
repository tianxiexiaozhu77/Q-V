#!/usr/bin/env python3
"""Run the released RadZero checkpoint on public NODE21 training-source images.

This is image-level, zero-shot nodule classification, not the private NODE21
challenge test or an experiment reported by the original RadZero paper.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from infer_batch import (
    ASSETS_DIR, MODEL_DIR, PROJECT_ROOT, CachedInference, checkpoint_identity,
    load_model, now, write_json,
)
from batch_metrics import compute_metrics
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


OUTPUTS = (ASSETS_DIR / "outputs").resolve()
DEFAULT_ROOT = PROJECT_ROOT.parent / "alpharad-assets/data_images/Node21"
KIND = "node21_official_radzero_zero_shot_v1"
SCOPE = ("Zero-shot image-level nodule classification on public NODE21 training-source "
         "images; no training, prompt tuning or threshold selection. This is not the "
         "private NODE21 challenge test, nor an original RadZero paper benchmark.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    return {name: file_sha(Path(__file__).with_name(name)) for name in (
        "infer_node21.py", "node21_data.py", "infer_batch.py", "infer.py", "batch_metrics.py")}


def shard(count, rank, world_size):
    return count * rank // world_size, count * (rank + 1) // world_size


def identity(args, spec):
    paths = [str(path) for path in spec["image_paths"]]
    labels = np.asarray(spec["labels"], dtype=np.int8)
    inventory = []
    for path in paths:
        stat = Path(path).stat()
        inventory.append([path, stat.st_size, stat.st_mtime_ns])
    return {
        "kind": KIND, "image_count": len(paths), "image_paths": paths,
        "labels": labels.tolist(), "class_names": list(spec["class_names"]),
        "prompts": list(spec["prompts"]), "gpus": args.gpus,
        "batch_size": args.batch_size, "workers": args.workers, "limit": args.limit,
        "data_root": str(args.data_root),
        "shards": [dict(zip(("start", "end"), shard(len(paths), rank, len(args.gpus))))
                   for rank in range(len(args.gpus))],
        "source_sha256": {str(path): file_sha(path) for path in spec["source_files"]},
        "image_inventory_sha256": digest(inventory),
        "image_inventory_method": "SHA256 of ordered absolute path, byte size and mtime_ns",
        "code_sha256": source_identity(), "checkpoint": checkpoint_identity(),
        "model_config_sha256": file_sha(MODEL_DIR / "config.json"),
        "preprocessor_sha256": file_sha(MODEL_DIR / "preprocessor_config.json"),
        "metadata": spec.get("metadata", {}), "evaluation_scope": SCOPE,
        "precision": "float32, TF32 disabled",
        "protocol": "node21_mha_per_image_minmax_rgb_author_518_single_prompt_v1",
    }


def prepare_run(args, spec):
    if len(spec["image_paths"]) < len(args.gpus):
        raise ValueError("Every GPU needs at least one image")
    value = identity(args, spec)
    fingerprint = digest(value)
    target = args.output / "run_manifest.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("Existing output belongs to different data, code or settings")
        return previous
    value.update(fingerprint=fingerprint, created_at=now(),
                 score_description="Sigmoid matching scores are not calibrated probabilities.",
                 map_description="Raw [N,1,37,37] similarity maps; bilinear resize to image_hw "
                                 "before sigmoid, align_corners=False.")
    write_json(target, value)
    return value


def chunks(directory, start, end):
    result, offset = [], start
    for path in sorted(directory.glob("batch_*.npz")):
        fields = path.stem.split("_")
        if len(fields) != 3:
            raise RuntimeError("Unexpected batch filename: " + str(path))
        first, stop = int(fields[1]), int(fields[2])
        if stop <= start or first >= end:
            continue
        if first != offset or stop <= first or stop > end:
            raise RuntimeError("Saved batch has a gap, overlap or crosses a shard: " + str(path))
        with np.load(path, allow_pickle=False) as saved:
            count = stop - first
            if not np.array_equal(saved["indices"], np.arange(first, stop)):
                raise RuntimeError("Invalid saved indices: " + str(path))
            for key, shape in {
                "logits": (count, 1), "similarity_scores": (count, 1, 37, 37),
                "image_hw": (count, 2),
            }.items():
                if saved[key].shape != shape or not np.isfinite(saved[key]).all():
                    raise RuntimeError("Invalid saved " + key + ": " + str(path))
            if (saved["image_hw"] <= 0).any():
                raise RuntimeError("Invalid original image dimensions")
        result.append(path)
        offset = stop
    return result, offset


def interruption(signum, _frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def worker(args):
    from node21_data import Node21Images, load_node21
    rank, gpu = args.worker_rank, args.gpus[args.worker_rank]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("Each worker must see only its assigned physical GPU")
    directory = args.output / "workers" / f"gpu{gpu}"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"status": "starting", "rank": rank, "physical_gpu": gpu,
              "cuda_visible_devices": str(gpu), "pid": os.getpid(), "started_at": now()}
    write_json(directory / "progress.json", status)
    try:
        spec = load_node21(root=args.data_root, limit=args.limit)
        manifest = json.loads((args.output / "run_manifest.json").read_text())
        if digest(identity(args, spec)) != manifest["fingerprint"]:
            raise RuntimeError("Worker configuration differs from the prepared manifest")
        start, end = shard(len(spec["image_paths"]), rank, len(args.gpus))
        _, offset = chunks(args.output / "batches", start, end)
        initial = offset
        timer = time.monotonic()
        status.update(status="running", start=start, end=end, total_images=end - start,
                      completed_images=offset - start)
        write_json(directory / "progress.json", status)
        if offset < end:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("This worker requires exactly one visible CUDA device")
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            status["gpu_name"] = torch.cuda.get_device_name(0)
            model, processor, tokenizer = load_model(torch.device("cuda"))
            images = Node21Images(spec["image_paths"], processor)
            loader_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
            if args.workers:
                loader_kwargs.update(multiprocessing_context="spawn",
                                     persistent_workers=True, prefetch_factor=2)
            loader = DataLoader(Subset(images, range(offset, end)), **loader_kwargs)
            with torch.inference_mode():
                inference = CachedInference(model, tokenizer, spec["prompts"])
                checked = False
                for pixels, sizes, indices in loader:
                    pixels = pixels.to(model.device, non_blocking=True)
                    if not checked:
                        checks = inference.verify_reference(pixels)
                        write_json(directory / "reference_equivalence.json",
                                   {"passed": True, "checks": checks,
                                    "checkpoint": manifest["checkpoint"]})
                        checked = True
                    logits, maps = inference(pixels)
                    first, stop = int(indices[0]), int(indices[-1]) + 1
                    if first != offset or not np.array_equal(indices.numpy(), np.arange(first, stop)):
                        raise RuntimeError("DataLoader returned unexpected image indices")
                    target = args.output / "batches" / f"batch_{first:06d}_{stop:06d}.npz"
                    temporary = target.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle, indices=indices.numpy(), logits=logits.float().cpu().numpy(),
                            similarity_scores=maps.float().cpu().numpy(), image_hw=sizes.numpy())
                    temporary.replace(target)
                    offset = stop
                    elapsed = time.monotonic() - timer
                    status.update(completed_images=offset - start, updated_at=now(),
                                  images_per_second=(offset - initial) / max(elapsed, .001),
                                  elapsed_seconds_this_session=elapsed)
                    write_json(directory / "progress.json", status)
                    if first == initial or stop == end or (stop - initial) % (args.batch_size * 10) == 0:
                        print(f"GPU {gpu}: {offset-start}/{end-start}, "
                              f"{status['images_per_second']:.2f} images/s", flush=True)
            del loader, inference, model, tokenizer, processor
            status["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        reference = directory / "reference_equivalence.json"
        if not reference.is_file() or not json.loads(reference.read_text()).get("passed"):
            raise RuntimeError("Missing successful original compute_logits equivalence check")
        status.update(status="complete", completed_at=now())
        write_json(directory / "progress.json", status)
        write_json(directory / "completion.json", status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), failed_at=now())
        write_json(directory / "progress.json", status)
        traceback.print_exc()
        raise
    finally:
        lock.close()


def export(args, spec):
    count = len(spec["image_paths"])
    saved, offset = chunks(args.output / "batches", 0, count)
    if offset != count:
        raise RuntimeError(f"Only {offset}/{count} predictions saved")
    parts = []
    for path in saved:
        with np.load(path, allow_pickle=False) as batch:
            parts.append(batch["logits"])
    logits = np.concatenate(parts)
    labels = np.asarray(spec["labels"], dtype=np.int8)
    if logits.shape != labels.shape:
        raise RuntimeError("Label/logit shapes differ")
    scores = 1 / (1 + np.exp(-logits.astype(np.float64)))
    class_name = spec["class_names"][0]
    target = args.output / "predictions.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path", f"label:{class_name}",
                         f"logit:{class_name}", f"score:{class_name}"])
        for index, path in enumerate(spec["image_paths"]):
            writer.writerow([index, path, int(labels[index, 0]),
                             float(logits[index, 0]), float(scores[index, 0])])
    temporary.replace(target)
    metrics = compute_metrics(spec, logits)
    metrics["evaluation_scope"] = SCOPE
    write_json(args.output / "metrics.json", metrics)
    return metrics


def stop_workers(processes):
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def worker_arguments(args, rank):
    return [sys.executable, str(Path(__file__).resolve()), "--worker-rank", str(rank),
            "--gpus", *map(str, args.gpus), "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--limit", str(args.limit),
            "--output", str(args.output), "--data-root", str(args.data_root)]


def coordinator(args, spec, manifest):
    processes = []
    started = time.monotonic()
    state = {"status": "starting", "started_at": now(), "pid": os.getpid(),
             "run_token": args.run_token, "gpus": args.gpus, "workers": {},
             "fingerprint": manifest["fingerprint"], "total_images": len(spec["image_paths"]),
             "evaluation_scope": SCOPE, "checkpoint": manifest["checkpoint"]}
    write_json(args.output / "progress.json", state)
    (args.output / "run.pid").write_text(str(os.getpid()) + "\n")
    try:
        for rank, gpu in enumerate(args.gpus):
            directory = args.output / "workers" / f"gpu{gpu}"
            directory.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
            with (directory / "run.log").open("a") as log:
                processes.append(subprocess.Popen(
                    worker_arguments(args, rank), env=environment, cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True))
        state.update(status="running", worker_pids=[process.pid for process in processes])
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Node21 workers failed with exit codes: {codes}")
            state["workers"] = {}
            for gpu in args.gpus:
                path = args.output / "workers" / f"gpu{gpu}" / "progress.json"
                if path.exists():
                    state["workers"][str(gpu)] = json.loads(path.read_text())
            state.update(updated_at=now(), completed_images=sum(
                value.get("completed_images", 0) for value in state["workers"].values()))
            write_json(args.output / "progress.json", state)
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
        if any(value.get("status") != "complete" for value in state["workers"].values()):
            raise RuntimeError("Workers exited without completion records")
        if digest(identity(args, spec)) != manifest["fingerprint"]:
            raise RuntimeError("Input data or inference code changed during the run")
        metrics = export(args, spec)
        state.update(status="complete", completed_at=now(), metrics=metrics,
                     elapsed_seconds=time.monotonic() - started,
                     predictions=str(args.output / "predictions.csv"),
                     map_batches=str(args.output / "batches"),
                     independent_audit="pending; run audit_node21.py")
        write_json(args.output / "summary.json", state)
        write_json(args.output / "completion.json", state)
        write_json(args.output / "progress.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        stop_workers(processes)
        state.update(status="failed", error=repr(error), failed_at=now())
        write_json(args.output / "progress.json", state)
        write_json(args.output / "failure.json", state)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="First N images; 0 evaluates all 4882")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-token", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus)
            or not set(args.gpus).issubset({0, 1}) or args.batch_size < 1
            or args.workers < 0 or args.limit < 0):
        parser.error("Use distinct GPUs from {0,1}, positive batch size and nonnegative workers/limit")
    if args.worker_rank is not None and not 0 <= args.worker_rank < len(args.gpus):
        parser.error("Invalid worker rank")
    args.output = args.output.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.output.parent != OUTPUTS or not args.output.name.startswith("node21_"):
        parser.error("Use a separate node21_* directory directly beneath " + str(OUTPUTS))
    signal.signal(signal.SIGTERM, interruption)
    signal.signal(signal.SIGINT, interruption)
    if args.worker_rank is not None:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / "owner.json"
    existing = list(args.output.iterdir())
    if existing and (not marker.is_file() or json.loads(marker.read_text()).get("kind") != KIND):
        parser.error("Refusing a nonempty output directory owned by another experiment")
    if args.lock_fd is not None:
        lock = os.fdopen(args.lock_fd, "a")
    else:
        lock = (args.output / ".run.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not marker.exists():
            write_json(marker, {"kind": KIND, "created_at": now()})
        from node21_data import load_node21
        spec = load_node21(root=args.data_root, limit=args.limit)
        manifest = prepare_run(args, spec)
        (args.output / "batches").mkdir(exist_ok=True)
        args.run_token = args.run_token or uuid.uuid4().hex
        if args.background:
            command = [sys.executable, str(Path(__file__).resolve()),
                       *[value for value in sys.argv[1:] if value != "--background"],
                       "--lock-fd", str(lock.fileno()), "--run-token", args.run_token]
            with (args.output / "run.log").open("a") as log:
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Background coordinator exited: {process.returncode}; see run.log")
                path = args.output / "progress.json"
                if path.exists():
                    status = json.loads(path.read_text())
                    if status.get("run_token") == args.run_token:
                        print(json.dumps({"status": status["status"], "pid": process.pid,
                                          "output": str(args.output),
                                          "log": str(args.output / "run.log")}), flush=True)
                        return
                time.sleep(.2)
            stop_workers([process])
            raise RuntimeError("Background coordinator did not acknowledge startup in 30 seconds")
        coordinator(args, spec, manifest)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
