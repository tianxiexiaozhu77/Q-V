#!/usr/bin/env python3
"""Evaluate a fixed prompt ensemble on the author test splits using separate GPUs."""

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

from infer_batch import (
    ASSETS_DIR, DATASETS, MODEL_DIR, PROJECT_ROOT, TestImages, checkpoint_identity,
    dataset_identity, load_model, now, write_json,
)
from batch_data import load_dataset
from batch_metrics import compute_metrics
from prompt_ensemble import PromptEnsembleInference, build_prompt_config
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


BASELINE = ASSETS_DIR / "outputs/public5_gpu0"
DEFAULT_OUTPUT = ASSETS_DIR / "outputs/public5_prompt_ensemble_gpu01"


def handle_termination(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping this run's workers")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def shard_range(count, rank, world_size):
    return count * rank // world_size, count * (rank + 1) // world_size


def source_identity():
    names = ("infer_prompt_ensemble.py", "prompt_ensemble.py", "infer_batch.py",
             "batch_data.py", "batch_metrics.py", "infer.py")
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names}


def prepare_run(args, specs):
    code_hashes = source_identity()
    identities = {}
    for name, spec in specs.items():
        paths, labels, base_fingerprint = dataset_identity(spec, args.limit)
        config = build_prompt_config(spec)
        shards = [dict(zip(("start", "end"), shard_range(len(paths), i, len(args.gpus))))
                  for i in range(len(args.gpus))]
        identity = {"baseline_fingerprint": base_fingerprint, "prompt_ensemble": config,
                    "shards": shards, "code_sha256": code_hashes}
        fingerprint = digest_json(identity)
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        (directory / "batches").mkdir(exist_ok=True)
        manifest = {
            "dataset": name, "fingerprint": fingerprint, "image_count": len(paths),
            "full_test_count": len(spec["image_paths"]), "class_names": spec["class_names"],
            "prompts": spec["prompts"], "prompt_ensemble": config, "shards": shards,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in spec["source_files"]},
            "metadata": spec["metadata"], "checkpoint": checkpoint_identity(),
            "code_sha256": code_hashes, "created_at": now(),
            "preprocessing": "Unchanged author evaluation preprocessing; Blip 518x518.",
            "precision": "float32; TF32 disabled",
            "map_description": "Equal-weight mean of raw per-prompt similarity maps [N,C,37,37]. "
                               "Bilinear resize before sigmoid. Ensemble map is descriptive; "
                               "not an exact explanation of the averaged nonlinear logits.",
            "template_logits_description": "[N,C,5], template 0 is the unmodified author prompt",
            "score_description": "Mean raw logits, then sigmoid; not calibrated clinical probabilities.",
        }
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["fingerprint"] != fingerprint:
                raise RuntimeError("Existing run uses a different configuration: " + str(directory))
        else:
            write_json(manifest_path, manifest)
        identities[name] = fingerprint
    run_identity = {"dataset_fingerprints": identities, "gpus": args.gpus,
                    "limit": args.limit, "code_sha256": code_hashes}
    path = args.output / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != digest_json(run_identity):
            raise RuntimeError("Cannot resume a different experiment in the same output directory")
    else:
        write_json(path, dict(run_identity, fingerprint=digest_json(run_identity),
                             created_at=now(), checkpoint=checkpoint_identity(),
                             selection_policy="Five prompts and equal weights fixed before test inference; "
                                              "no test-label-based prompt or weight selection."))


def read_chunks(directory, start, end, class_count):
    """Validate and return the contiguous completed prefix for one worker's range."""
    result, offset = [], start
    for path in sorted(directory.glob("batch_*.npz")):
        fields = path.stem.split("_")
        if len(fields) != 3:
            raise RuntimeError("Unexpected chunk name: " + str(path))
        first, stop = int(fields[1]), int(fields[2])
        if stop <= start or first >= end:
            continue
        if first < start or stop > end or first != offset:
            raise RuntimeError("Saved chunk crosses a shard or leaves an index gap: " + str(path))
        with np.load(path, allow_pickle=False) as chunk:
            count = stop - first
            if count <= 0 or not np.array_equal(chunk["indices"], np.arange(first, stop)):
                raise RuntimeError("Invalid chunk indices: " + str(path))
            required = {
                "logits": (count, class_count),
                "template_logits": (count, class_count, 5),
                "similarity_scores": (count, class_count, 37, 37),
                "image_hw": (count, 2),
            }
            for key, shape in required.items():
                if chunk[key].shape != shape or not np.isfinite(chunk[key]).all():
                    raise RuntimeError("Invalid saved " + key + ": " + str(path))
            np.testing.assert_allclose(chunk["logits"], chunk["template_logits"].mean(axis=-1),
                                       atol=2e-5, rtol=1e-4)
        result.append(path)
        offset = stop
    return result, offset


def worker(args):
    rank = args.worker_rank
    physical_gpu = args.gpus[rank]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("Worker must be launched with its assigned physical GPU only")
    directory = args.output / "workers" / ("gpu" + str(physical_gpu))
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (directory / "run.pid").write_text(str(os.getpid()) + "\n")
    status = {"status": "starting", "started_at": now(), "pid": os.getpid(), "rank": rank,
              "cuda_visible_devices": str(physical_gpu), "datasets": {},
              "batch_size": args.batch_size, "workers": args.workers}
    status_path = directory / "progress.json"
    write_json(status_path, status)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Each worker requires exactly its one assigned visible GPU")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        status.update(status="running", gpu_name=torch.cuda.get_device_name(0))
        write_json(status_path, status)
        model, processor, tokenizer = load_model(torch.device("cuda"))
        with torch.inference_mode():
            for name in args.datasets:
                spec = load_dataset(name, ASSETS_DIR / "data")
                manifest = json.loads((args.output / name / "manifest.json").read_text())
                if manifest["code_sha256"] != source_identity():
                    raise RuntimeError("Inference source changed after this run was configured")
                config = build_prompt_config(spec)
                paths, labels, base_fingerprint = dataset_identity(spec, args.limit)
                expected = digest_json({
                    "baseline_fingerprint": base_fingerprint, "prompt_ensemble": config,
                    "shards": manifest["shards"], "code_sha256": source_identity(),
                })
                if expected != manifest["fingerprint"]:
                    raise RuntimeError("Worker's data or prompts differ from the prepared run")
                start, end = shard_range(len(paths), rank, len(args.gpus))
                batch_dir = args.output / name / "batches"
                _, offset = read_chunks(batch_dir, start, end, len(spec["class_names"]))
                initial_offset = offset
                timer = time.monotonic()
                entry = {"status": "running", "start": start, "end": end,
                         "completed_images": offset - start, "total_images": end - start,
                         "updated_at": now()}
                status["datasets"][name] = entry
                write_json(status_path, status)
                print(f"GPU {physical_gpu} {name}: [{start},{end}), resume at {offset}", flush=True)
                if offset < end:
                    images = TestImages(paths, ASSETS_DIR / "data", processor,
                                        name in ("SIIM", "RSNA"))
                    loader_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
                    if args.workers:
                        loader_kwargs.update(multiprocessing_context="spawn",
                                             persistent_workers=True, prefetch_factor=2)
                    loader = DataLoader(Subset(images, range(offset, end)), **loader_kwargs)
                    infer = PromptEnsembleInference(model, tokenizer, config)
                    checked = False
                    for pixels, sizes, indices in loader:
                        pixels = pixels.to(model.device, non_blocking=True)
                        if not checked:
                            check = infer.verify_reference(pixels)
                            write_json(directory / (name + "_reference_equivalence.json"),
                                       {"passed": True, "checks": check})
                            checked = True
                        outputs = infer(pixels)
                        arrays = {key: outputs[key].float().cpu().numpy()
                                  for key in ("logits", "template_logits", "similarity_scores")}
                        first, stop = int(indices[0]), int(indices[-1]) + 1
                        target = batch_dir / f"batch_{first:06d}_{stop:06d}.npz"
                        temporary = target.with_suffix(".npz.tmp")
                        with temporary.open("wb") as handle:
                            np.savez_compressed(handle, indices=indices.numpy(),
                                                image_hw=sizes.numpy(), **arrays)
                        temporary.replace(target)
                        offset = stop
                        elapsed = time.monotonic() - timer
                        entry.update(completed_images=offset - start, updated_at=now(),
                                     images_per_second=(offset - initial_offset) / max(elapsed, .001),
                                     elapsed_seconds_this_session=elapsed)
                        write_json(status_path, status)
                        if first == initial_offset or stop == end or (
                                stop - initial_offset) % (args.batch_size * 10) == 0:
                            print(f"GPU {physical_gpu} {name}: {offset-start}/{end-start}, "
                                  f"{entry['images_per_second']:.2f} images/s", flush=True)
                    del loader, infer
                entry.update(status="complete", updated_at=now())
                status["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                write_json(status_path, status)
        status.update(status="complete", completed_at=now())
        write_json(status_path, status)
        print("WORKER COMPLETE", flush=True)
    except BaseException as error:
        status.update(status="failed", error=repr(error), failed_at=now())
        write_json(status_path, status)
        traceback.print_exc()
        raise
    finally:
        lock.close()


def export_dataset(args, name, spec):
    paths = spec["image_paths"][:args.limit] if args.limit else spec["image_paths"]
    labels = spec["labels"][:len(paths)]
    directory = args.output / name
    chunks, offset = read_chunks(directory / "batches", 0, len(paths), len(spec["class_names"]))
    if offset != len(paths):
        raise RuntimeError(f"{name}: only {offset}/{len(paths)} predictions saved")
    logits_parts = []
    for path in chunks:
        with np.load(path, allow_pickle=False) as chunk:
            logits_parts.append(chunk["logits"])
    logits = np.concatenate(logits_parts)
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    destination = directory / "predictions.csv"
    temporary = destination.with_suffix(".csv.tmp")
    classes = spec["class_names"]
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path"] + ["label:" + c for c in classes] +
                        ["logit:" + c for c in classes] + ["score:" + c for c in classes])
        for index, path in enumerate(paths):
            writer.writerow([index, path] + labels[index].tolist() + logits[index].tolist() +
                            probabilities[index].tolist())
    temporary.replace(destination)
    metrics = compute_metrics(dict(spec, labels=labels, image_paths=paths), logits)
    write_json(directory / "metrics.json", metrics)
    result = {"status": "complete", "completed_images": len(paths), "total_images": len(paths),
              "classes": len(classes), "updated_at": now(), "metrics": metrics,
              "predictions": str(destination), "map_batches": str(directory / "batches")}
    write_json(directory / "progress.json", result)
    return result


def coordinator(args):
    lock = (args.output / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "run.pid").write_text(str(os.getpid()) + "\n")
    state = {"status": "starting", "started_at": now(), "pid": os.getpid(),
             "gpus": args.gpus, "limit": args.limit, "datasets": {}}
    workers = []
    try:
        specs = {name: load_dataset(name, ASSETS_DIR / "data") for name in args.datasets}
        prepare_run(args, specs)
        state.update(status="running", checkpoint=checkpoint_identity(),
                     total_images=sum(min(args.limit, len(spec["image_paths"])) if args.limit
                                      else len(spec["image_paths"]) for spec in specs.values()))
        write_json(args.output / "progress.json", state)
        for rank, gpu in enumerate(args.gpus):
            directory = args.output / "workers" / ("gpu" + str(gpu))
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-rank", str(rank),
                       "--gpus"] + [str(i) for i in args.gpus] + [
                           "--datasets", *args.datasets, "--batch-size", str(args.batch_size),
                           "--workers", str(args.workers), "--limit", str(args.limit),
                           "--output", str(args.output)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
            with (directory / "run.log").open("a") as log:
                process = subprocess.Popen(command, env=environment, cwd=PROJECT_ROOT,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
            workers.append(process)
        state["worker_pids"] = [process.pid for process in workers]
        write_json(args.output / "progress.json", state)
        while any(process.poll() is None for process in workers):
            failed = [(args.gpus[i], process.returncode) for i, process in enumerate(workers)
                      if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("GPU workers failed: " + repr(failed))
            state["workers"] = {}
            for gpu in args.gpus:
                path = args.output / "workers" / ("gpu" + str(gpu)) / "progress.json"
                if path.exists():
                    state["workers"][str(gpu)] = json.loads(path.read_text())
            state["updated_at"] = now()
            write_json(args.output / "progress.json", state)
            time.sleep(2)
        if any(process.returncode != 0 for process in workers):
            raise RuntimeError("A GPU worker exited unsuccessfully")
        state["workers"] = {
            str(gpu): json.loads((args.output / "workers" / ("gpu" + str(gpu)) /
                                 "progress.json").read_text())
            for gpu in args.gpus
        }
        for name, spec in specs.items():
            state["datasets"][name] = export_dataset(args, name, spec)
            write_json(args.output / "progress.json", state)
        state.update(status="inference_complete", completed_at=now())
        write_json(args.output / "progress.json", state)
        command = [sys.executable, str(Path(__file__).with_name("summarize_prompt_ensemble.py")),
                   "--run-dir", str(args.output), "--baseline-dir", str(BASELINE),
                   "--datasets", *args.datasets, "--limit", str(args.limit)]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        validation = json.loads((args.output / "validation.json").read_text())
        if validation.get("status") != "passed":
            raise RuntimeError("Final comparison audit did not pass")
        state.update(status="complete", validated_at=now())
        write_json(args.output / "progress.json", state)
        print(json.dumps({"status": "complete", "output": str(args.output),
                          "gpus": args.gpus, "images": state["total_images"]}), flush=True)
    except BaseException as error:
        for process in workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in workers:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        state.update(status="failed", error=repr(error), failed_at=now())
        write_json(args.output / "progress.json", state)
        traceback.print_exc()
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0
            or args.batch_size < 1 or args.workers < 0 or args.limit < 0):
        parser.error("Require distinct nonnegative GPUs, positive batch size, nonnegative workers/limit")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Datasets must not be repeated")
    if args.limit and args.limit < len(args.gpus):
        parser.error("--limit must give every selected GPU at least one image per dataset")
    if args.worker_rank is not None and not 0 <= args.worker_rank < len(args.gpus):
        parser.error("Invalid internal worker rank")
    args.output = args.output.expanduser().resolve()
    for protected in (MODEL_DIR.resolve(), BASELINE.resolve()):
        if args.output == protected or protected in args.output.parents or args.output in protected.parents:
            parser.error("Output overlaps the original checkpoint or baseline directory")
    args.output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_termination)
    if args.worker_rank is not None:
        worker(args)
    elif args.background:
        command = [sys.executable, str(Path(__file__).resolve())] + [
            value for value in sys.argv[1:] if value != "--background"]
        with (args.output / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(args.output / "run.log"),
                          "output": str(args.output)}))
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
