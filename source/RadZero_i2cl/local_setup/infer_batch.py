#!/usr/bin/env python3
"""Run the five public RadZero test sets with the released local HF checkpoint."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from infer import ASSETS_DIR, MODEL_DIR, PROJECT_ROOT, configure_offline, read_local_config

configure_offline()

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


DATASETS = ("ChestXDet10", "SIIM", "RSNA", "OpenI", "ChestXray14")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def checkpoint_identity():
    manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text())
    expected = next(x["hash"] for x in manifest["files"] if x["path"] == "model.safetensors")
    digest = hashlib.sha256()
    with (MODEL_DIR / "model.safetensors").open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError("Checkpoint SHA256 differs from the verified official download")
    return {"repo": manifest["repo"], "revision": manifest["revision"], "weight_sha256": actual}


class TestImages(Dataset):
    def __init__(self, paths, root, processor, dicom):
        self.paths, self.root, self.processor, self.dicom = paths, root, processor, dicom

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import cv2
        cv2.setNumThreads(0)
        path = self.root / self.paths[index]
        if self.dicom:
            import pydicom
            dcm = pydicom.dcmread(path)
            pixels = dcm.pixel_array
            maximum = float(pixels.max())
            pixels = cv2.convertScaleAbs(pixels, alpha=255.0 / maximum) if maximum else np.zeros_like(pixels, dtype=np.uint8)
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                pixels = cv2.bitwise_not(pixels)
            image = Image.fromarray(pixels).convert("RGB")
        else:
            with Image.open(path) as original:
                pixels = cv2.normalize(np.array(original), None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            image = Image.fromarray(pixels)
        size = (image.height, image.width)
        tensor = self.processor(image, return_tensors="pt")["pixel_values"][0]
        return tensor, torch.tensor(size, dtype=torch.int32), index


class CachedInference:
    """Cache unchanged text embeddings; retain the released attention and logits."""
    def __init__(self, model, tokenizer, prompts):
        self.model = model
        self.loss = model.loss_fns["KeyPhraseAlignmentLoss"]
        self.tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        split = [{key: value[i:i + 1] for key, value in self.tokens.items() if key in ("input_ids", "attention_mask")}
                 for i in range(len(prompts))]
        self.features, _ = self.loss.compute_text_features(split, model.forward_text_model, ddp_gather=False)
        self.class_count = len(prompts)

    def __call__(self, pixels):
        vision = self.model.forward_vision_model(pixels)["vision_tokens"]
        if self.loss.layer_norm is not None:
            vision = self.loss.layer_norm(vision)
        if not self.loss.use_vision_cls_token:
            vision = vision[:, 1:]
        logits, attention = self.loss.compute_t2i_logits(self.features, vision, need_attn_weights=True)
        # The released loss squeezes singleton dimensions. Restore [class,batch]
        # before transposing, including binary tasks and the final one-image batch.
        logits = logits.reshape(self.class_count, len(pixels)).T / self.loss.loss_temperature.exp()
        scores = torch.stack(attention).mean(dim=0)
        if self.loss.use_vision_cls_token:
            scores = scores[:, :, 1:]
        side = int(scores.shape[-1] ** 0.5)
        if side * side != scores.shape[-1]:
            raise RuntimeError(f"Non-square patch map: {scores.shape}")
        maps = scores.reshape(len(pixels), self.class_count, side, side)
        if not torch.isfinite(logits).all() or not torch.isfinite(maps).all():
            raise RuntimeError("Non-finite logits or similarity maps")
        return logits, maps

    def verify_reference(self, pixels):
        checks = []
        for count in sorted({1, min(2, len(pixels))}):
            subset = pixels[:count]
            logits, maps = self(subset)
            reference = self.model.compute_logits(subset, [self.tokens])
            expected_logits = reference["logits"].reshape(count, self.class_count)
            expected_maps = reference["similarity_scores"].reshape_as(maps)
            torch.testing.assert_close(logits, expected_logits, atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(maps, expected_maps, atol=2e-5, rtol=1e-4)
            checks.append({"batch_size": count, "classes": self.class_count,
                           "max_logit_error": float((logits - expected_logits).abs().max()),
                           "max_map_error": float((maps - expected_maps).abs().max())})
        return checks


def load_model(device):
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    config_dict = read_local_config()
    cls = get_class_from_dynamic_module(config_dict["auto_map"]["AutoConfig"], str(MODEL_DIR), local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL_DIR), config=cls.from_dict(config_dict),
                                     trust_remote_code=True, local_files_only=True,
                                     torch_dtype=torch.float32).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    return model, processor, tokenizer


def dataset_identity(spec, limit):
    paths = spec["image_paths"][:limit] if limit else spec["image_paths"]
    labels = np.asarray(spec["labels"][:len(paths)], dtype=np.int8)
    identity = {"paths": paths, "classes": spec["class_names"], "prompts": spec["prompts"],
                "labels": labels.tolist(), "model": str(MODEL_DIR),
                "checkpoint": checkpoint_identity(),
                "model_config_sha256": hashlib.sha256((MODEL_DIR / "config.json").read_bytes()).hexdigest(),
                "protocol": "official-test-preprocessing-fp32-v1"}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return paths, labels, digest


def completed_chunks(directory, expected_count, classes):
    chunks, offset = [], 0
    for path in sorted(directory.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as chunk:
            indices = chunk["indices"]
            if not np.array_equal(indices, np.arange(offset, offset + len(indices))):
                raise RuntimeError(f"Non-contiguous saved indices in {path}")
            if chunk["logits"].shape != (len(indices), classes) or not np.isfinite(chunk["logits"]).all():
                raise RuntimeError(f"Invalid saved logits in {path}")
            if chunk["similarity_scores"].shape != (len(indices), classes, 37, 37) or not np.isfinite(chunk["similarity_scores"]).all():
                raise RuntimeError(f"Invalid saved maps in {path}")
            offset += len(indices)
        chunks.append(path)
    if offset > expected_count:
        raise RuntimeError("Saved result is longer than the requested test set")
    return chunks, offset


def run_dataset(name, spec, args, model, processor, tokenizer, overall):
    from batch_metrics import compute_metrics
    paths, labels, fingerprint = dataset_identity(spec, args.limit)
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    batches = directory / "batches"
    batches.mkdir(exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError(f"Existing outputs use different data/configuration: {directory}")
    else:
        source_hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in spec["source_files"]}
        write_json(manifest_path, {"dataset": name, "fingerprint": fingerprint, "image_count": len(paths),
                                  "full_test_count": len(spec["image_paths"]), "class_names": spec["class_names"],
                                  "prompts": spec["prompts"], "source_sha256": source_hashes,
                                  "metadata": spec.get("metadata", {}),
                                  "map_description": "Raw attention similarity scores [N,C,37,37]. For original-resolution heatmaps, bilinear interpolate these scores to image_hw before sigmoid, align_corners=False.",
                                  "score_description": "Sigmoid matching scores are not calibrated clinical probabilities.",
                                  "preprocessing": "Official evaluation PNG minmax / DICOM max scaling and inversion; Blip 518x518.",
                                  "precision": "float32, TF32 disabled", "created_at": now()})
    chunk_paths, offset = completed_chunks(batches, len(paths), len(spec["prompts"]))
    print(f"{name}: {offset}/{len(paths)} already saved, {len(spec['prompts'])} prompts", flush=True)
    started = time.monotonic()
    initial_offset = offset
    reference_checks = []
    if offset < len(paths):
        dataset = TestImages(paths, ASSETS_DIR / "data", processor, name in ("SIIM", "RSNA"))
        loader_kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        if args.workers:
            loader_kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(Subset(dataset, range(offset, len(paths))), **loader_kwargs)
        infer = CachedInference(model, tokenizer, spec["prompts"])
        for pixels, sizes, indices in loader:
            pixels = pixels.to(model.device, non_blocking=True)
            if not reference_checks:
                reference_checks = infer.verify_reference(pixels)
                write_json(directory / "reference_equivalence.json", {"passed": True, "checks": reference_checks})
            logits, maps = infer(pixels)
            logits_np, maps_np = logits.float().cpu().numpy(), maps.float().cpu().numpy()
            first, end = int(indices[0]), int(indices[-1]) + 1
            output = batches / f"batch_{first:06d}_{end:06d}.npz"
            temporary = output.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, indices=indices.numpy(), logits=logits_np,
                                    similarity_scores=maps_np, image_hw=sizes.numpy())
            temporary.replace(output)
            chunk_paths.append(output)
            offset = end
            elapsed = time.monotonic() - started
            status = {"status": "running", "dataset": name, "completed_images": offset,
                      "total_images": len(paths), "images_per_second": (offset-initial_offset)/max(elapsed, .001),
                      "elapsed_seconds_this_session": elapsed, "updated_at": now()}
            write_json(directory / "progress.json", status)
            overall["datasets"][name] = status
            write_json(args.output / "progress.json", overall)
            if first == initial_offset or end == len(paths) or (end-initial_offset) % (args.batch_size * 10) == 0:
                print(f"{name}: {end}/{len(paths)} ({status['images_per_second']:.2f} images/s)", flush=True)
        del loader, infer
    arrays = []
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as chunk:
            arrays.append(chunk["logits"])
    logits = np.concatenate(arrays)
    if logits.shape != labels.shape:
        raise RuntimeError(f"Label/score shape mismatch: {logits.shape} vs {labels.shape}")
    prediction_path = directory / "predictions.csv"
    temporary = prediction_path.with_suffix(".csv.tmp")
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image_path"] + [f"label:{c}" for c in spec["class_names"]]
                        + [f"logit:{c}" for c in spec["class_names"]] + [f"score:{c}" for c in spec["class_names"]])
        for i, path in enumerate(paths):
            writer.writerow([i, path] + labels[i].tolist() + logits[i].tolist() + probabilities[i].tolist())
    temporary.replace(prediction_path)
    metrics_spec = dict(spec, labels=labels, image_paths=paths)
    metrics = compute_metrics(metrics_spec, logits)
    write_json(directory / "metrics.json", metrics)
    status = {"status": "complete", "completed_images": len(paths), "total_images": len(paths),
              "classes": len(spec["prompts"]), "updated_at": now(), "metrics": metrics,
              "predictions": str(prediction_path), "map_batches": str(batches)}
    write_json(directory / "progress.json", status)
    overall["datasets"][name] = status
    write_json(args.output / "progress.json", overall)
    print(f"COMPLETE {name}: {len(paths)} images; metrics={json.dumps(metrics)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="First N test images per dataset; 0 means all")
    parser.add_argument("--output", type=Path, default=ASSETS_DIR / "outputs/public5_gpu0")
    parser.add_argument("--background", action="store_true", help="Launch a detached resumable run and report its PID")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.limit < 0:
        parser.error("batch-size must be positive; workers and limit must be nonnegative")
    args.output = args.output.expanduser().resolve()
    if args.output == MODEL_DIR or MODEL_DIR in args.output.parents:
        parser.error("Outputs must not be placed inside the original checkpoint snapshot")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != "--background"]
        with (args.output / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(args.output / "run.log"), "output": str(args.output)}))
        return
    lock = (args.output / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.output / "run.pid").write_text(str(os.getpid()) + "\n")
    overall = {"status": "starting", "started_at": now(), "pid": os.getpid(), "datasets": {}}
    write_json(args.output / "progress.json", overall)
    try:
        from batch_data import load_dataset
        specs = {name: load_dataset(name, ASSETS_DIR / "data") for name in args.datasets}
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        overall.update(status="running", device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       gpu_name=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                       checkpoint=checkpoint_identity(),
                       batch_size=args.batch_size, workers=args.workers, limit=args.limit,
                       total_images=sum(min(args.limit, len(s["image_paths"])) if args.limit else len(s["image_paths"]) for s in specs.values()))
        write_json(args.output / "progress.json", overall)
        print(json.dumps(overall), flush=True)
        model, processor, tokenizer = load_model(torch.device(args.device))
        with torch.inference_mode():
            for name, spec in specs.items():
                run_dataset(name, spec, args, model, processor, tokenizer, overall)
        overall.update(status="complete", completed_at=now())
        write_json(args.output / "progress.json", overall)
        write_json(args.output / "summary.json", overall)
        print("ALL DATASETS COMPLETE", flush=True)
    except BaseException as exc:
        overall.update(status="failed", error=repr(exc), failed_at=now())
        write_json(args.output / "progress.json", overall)
        traceback.print_exc()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
