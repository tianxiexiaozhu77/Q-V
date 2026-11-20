#!/usr/bin/env python3
"""Frozen RadZero features, DDP residual-adapter training, held-out Node21 evaluation."""

import argparse
import csv
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

from infer_batch import ASSETS_DIR, MODEL_DIR, checkpoint_identity, load_model, now, write_json
from batch_metrics import compute_metrics
from node21_adapter_data import load_split
from node21_data import Node21Images, PREPROCESSING_PROTOCOL
from siim_adapter_model import build_head
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset


SPLITS = ("train", "valid", "test")
PROMPT = "There is Lung Nodule"
BASELINE_DIRECTORY = ASSETS_DIR / "outputs/node21_zeroshot_gpu01"
BASELINE_PREDICTIONS = BASELINE_DIRECTORY / "predictions.csv"
LOCAL_SOURCE_FILES = (
    "train_node21_adapter.py", "siim_adapter_model.py", "node21_adapter_data.py",
    "node21_data.py", "infer_batch.py", "infer.py", "batch_metrics.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {name: sha256(Path(__file__).with_name(name)) for name in LOCAL_SOURCE_FILES}


def model_source_hashes():
    return {str(path.relative_to(MODEL_DIR)): sha256(path)
            for path in sorted(MODEL_DIR.rglob("*.py"))}


def tokenizer_hashes():
    names = ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt")
    return {name: sha256(MODEL_DIR / name) for name in names}


def tensor_digest(module):
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def partition(n, rank, world):
    return n * rank // world, n * (rank + 1) // world


def metrics(spec, logits):
    return compute_metrics(dict(spec, name="Node21"), np.asarray(logits).reshape(-1, 1))


def save_predictions(path, spec, logits):
    values = np.asarray(logits).reshape(-1)
    if len(values) != len(spec["image_paths"]) or not np.isfinite(values).all():
        raise RuntimeError("Prediction count or finite-value check failed")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "image_path", "label:Nodule",
                         "logit:Nodule", "score:Nodule"])
        for index, image_path in enumerate(spec["image_paths"]):
            value = float(values[index])
            writer.writerow([index, image_path, int(spec["labels"][index, 0]), value,
                             1.0 / (1.0 + np.exp(-value))])
    temporary.replace(path)


class TokenCache(Dataset):
    def __init__(self, directory, name, labels):
        self.path = Path(directory) / (name + "_tokens.npy")
        self.labels = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        self.tokens = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.tokens is None:
            self.tokens = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(self.tokens[index], copy=True)), torch.from_numpy(
            self.labels[index].copy()), index


def loader_options(args, batch_size, workers=None):
    count = args.workers if workers is None else workers
    result = dict(batch_size=batch_size, num_workers=count, pin_memory=True)
    if count:
        result.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    return result


def select_specs(args):
    result = {}
    for name in SPLITS:
        original = load_split(args.splits_dir, name)
        limit = getattr(args, "limit_" + name)
        count = min(limit, len(original["image_paths"])) if limit else len(original["image_paths"])
        spec = dict(original)
        for key in ("image_paths", "labels", "ids", "patient_ids", "study_ids", "group_ids"):
            if key in spec:
                spec[key] = spec[key][:count]
        if count < dist.get_world_size() or len(np.unique(spec["labels"])) != 2:
            raise RuntimeError(f"{name}: require at least one image per GPU and both binary classes")
        if spec["prompts"] != [PROMPT] or spec["class_names"] != ["Nodule"]:
            raise RuntimeError("Node21 labels and prompt differ from the fixed baseline protocol")
        result[name] = spec
    for index, name in enumerate(SPLITS):
        for other in SPLITS[index + 1:]:
            if set(result[name]["image_paths"]) & set(result[other]["image_paths"]):
                raise RuntimeError(f"Overlapping images between {name} and {other}")
    return result


def prepare_cache(args, specs, rank, world):
    identity = {
        "version": "node21-pre-layernorm-token-cache-fp32-v1",
        "checkpoint": checkpoint_identity(),
        "preprocessor_sha256": sha256(MODEL_DIR / "preprocessor_config.json"),
        "model_config_sha256": sha256(MODEL_DIR / "config.json"),
        "split_manifest_sha256": sha256(args.splits_dir / "split_manifest.json"),
        "counts": {name: len(spec["image_paths"]) for name, spec in specs.items()},
        "paths_and_labels": {name: fingerprint({
            "paths": spec["image_paths"], "labels": spec["labels"].tolist()
        }) for name, spec in specs.items()},
        "world_size": world, "shape_per_image": [1370, 768], "dtype": "float32",
        "extractor_sha256": sha256(Path(__file__)),
        "adapter_code_sha256": sha256(Path(__file__).with_name("siim_adapter_model.py")),
        "code_sha256": source_hashes(),
        "model_source_sha256": model_source_hashes(),
        "tokenizer_sha256": tokenizer_hashes(),
        "split_csv_sha256": {name: sha256(args.splits_dir / (name + ".csv")) for name in SPLITS},
        "prompt": PROMPT,
        "preprocessing": PREPROCESSING_PROTOCOL,
    }
    identity["fingerprint"] = fingerprint(identity)
    if rank == 0:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        path = args.cache_dir / "manifest.json"
        existing_progress = any(args.cache_dir.glob("rank*_progress.json"))
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["fingerprint"] != identity["fingerprint"]:
                raise RuntimeError("Feature cache belongs to another configuration")
            existing_progress = existing_progress or previous.get("status") == "complete"
        else:
            if existing_progress or any(args.cache_dir.glob("*.npy")):
                raise RuntimeError("Cache data exists without its identity manifest")
            write_json(path, dict(identity, status="extracting", created_at=now()))
        for name, spec in specs.items():
            count = len(spec["image_paths"])
            for suffix, shape in (("tokens", (count, 1370, 768)), ("baseline_logits", (count, 1))):
                path = args.cache_dir / (name + "_" + suffix + ".npy")
                if not path.exists():
                    if existing_progress:
                        raise RuntimeError("Cannot resume: a recorded cache file is missing: " + str(path))
                    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
                    array.flush()
                    del array
                else:
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if array.shape != shape or array.dtype != np.float32:
                        raise RuntimeError("Invalid feature-cache shape or dtype: " + str(path))
                    del array
    dist.barrier()
    return identity


def extract_features(args, specs, model, processor, head, device, rank, world, identity):
    directory = args.output / ("rank" + str(rank))
    directory.mkdir(exist_ok=True)
    progress_path = args.cache_dir / ("rank" + str(rank) + "_progress.json")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != identity["fingerprint"]:
            raise RuntimeError("Cache progress belongs to another configuration")
    else:
        progress = {"fingerprint": identity["fingerprint"], "status": "extracting",
                    "rank": rank, "started_at": now(), "splits": {}}
    for name, spec in specs.items():
        count = len(spec["image_paths"])
        start, end = partition(count, rank, world)
        offset = progress["splits"].get(name, {}).get("end_index", start)
        if not start <= offset <= end:
            raise RuntimeError("Invalid extraction resume offset")
        token_file = np.load(args.cache_dir / (name + "_tokens.npy"), mmap_mode="r+")
        baseline_file = np.load(args.cache_dir / (name + "_baseline_logits.npy"), mmap_mode="r+")
        timer = time.monotonic()
        initial = offset
        if offset < end:
            images = Node21Images(spec["image_paths"], processor)
            loader = DataLoader(Subset(images, range(offset, end)),
                                **loader_options(args, args.extract_batch_size))
            verified = False
            with torch.no_grad():
                for pixels, image_hw, indices in loader:
                    pixels = pixels.to(device, non_blocking=True)
                    tokens = model.forward_vision_model(pixels)["vision_tokens"]
                    original = head(tokens)["logits"]
                    if tokens.shape[1:] != (1370, 768) or not torch.isfinite(tokens).all():
                        raise RuntimeError("Invalid frozen image features")
                    if not verified:
                        checks = []
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
                        encoded = tokenizer(spec["prompts"], padding=True,
                                            return_tensors="pt").to(device)
                        for n in sorted({1, min(2, len(pixels))}):
                            reference = model.compute_logits(pixels[:n], [encoded])
                            # Compare the adapter with the official API at the
                            # same encoder batch size. FP32 encoder kernels have
                            # small, separate batch-size rounding differences.
                            same_batch_tokens = model.forward_vision_model(pixels[:n])["vision_tokens"]
                            actual = head(same_batch_tokens, return_maps=True)
                            expected = reference["logits"].reshape(n, 1)
                            maps = reference["similarity_scores"].reshape_as(actual["similarity_scores"])
                            torch.testing.assert_close(actual["logits"], expected, atol=2e-5, rtol=1e-4)
                            torch.testing.assert_close(actual["similarity_scores"], maps, atol=2e-5, rtol=1e-4)
                            cached = head(tokens[:n], return_maps=True)
                            torch.testing.assert_close(cached["logits"], expected, atol=2e-5, rtol=1e-4)
                            torch.testing.assert_close(cached["similarity_scores"], maps, atol=1e-4, rtol=1e-4)
                            checks.append({"batch_size": n,
                                           "max_logit_error": float((actual["logits"]-expected).abs().max()),
                                           "max_map_error": float((actual["similarity_scores"]-maps).abs().max()),
                                           "cached_batch_max_logit_error": float((cached["logits"]-expected).abs().max()),
                                           "cached_batch_max_map_error": float((cached["similarity_scores"]-maps).abs().max()),
                                           "cached_map_atol": 1e-4, "cached_map_rtol": 1e-4})
                        write_json(directory / (name + "_identity_reference.json"),
                                   {"passed": True, "checks": checks})
                        verified = True
                    first, stop = int(indices[0]), int(indices[-1]) + 1
                    token_file[first:stop] = tokens.float().cpu().numpy()
                    baseline_file[first:stop] = original.float().cpu().numpy()
                    token_file.flush()
                    baseline_file.flush()
                    offset = stop
                    progress["splits"][name] = {
                        "start": start, "end": end, "end_index": offset,
                        "completed_images": offset-start, "total_images": end-start,
                        "images_per_second": (offset-initial)/max(time.monotonic()-timer, .001),
                    }
                    progress["updated_at"] = now()
                    write_json(progress_path, progress)
                    if first == initial or stop == end or (stop-initial) % (args.extract_batch_size*10) == 0:
                        print(f"rank{rank} CACHE {name}: {offset-start}/{end-start}", flush=True)
            del loader
        del token_file, baseline_file
    progress.update(status="complete", completed_at=now())
    write_json(progress_path, progress)
    dist.barrier()
    if rank == 0:
        for name, spec in specs.items():
            baseline = np.load(args.cache_dir / (name + "_baseline_logits.npy"), allow_pickle=False)
            if baseline.shape != spec["labels"].shape or not np.isfinite(baseline).all():
                raise RuntimeError("Incomplete cached baseline scores")
        manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        manifest.update(status="complete", completed_at=now(),
                        total_images=sum(len(s["image_paths"]) for s in specs.values()),
                        tokens_bytes=sum((args.cache_dir/(n+"_tokens.npy")).stat().st_size for n in SPLITS))
        write_json(args.cache_dir / "manifest.json", manifest)
    dist.barrier()


def predict(head, dataset, args, device, rank, world):
    start, end = partition(len(dataset), rank, world)
    loader = DataLoader(Subset(dataset, range(start, end)),
                        **loader_options(args, args.batch_size))
    indices, values = [], []
    head.eval()
    with torch.no_grad():
        for tokens, labels, rows in loader:
            output = head(tokens.to(device, non_blocking=True))["logits"]
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite evaluation logits")
            indices.extend(rows.tolist())
            values.extend(output.float().cpu().flatten().tolist())
    gathered = [None] * world
    dist.all_gather_object(gathered, (indices, values))
    result = np.empty((len(dataset), 1), dtype=np.float64)
    seen = []
    for rows, scores in gathered:
        result[rows, 0] = scores
        seen.extend(rows)
    if sorted(seen) != list(range(len(dataset))):
        raise RuntimeError("Evaluation shards omit or duplicate images")
    del loader
    return result


def compare_original_test(spec, cached_logits, limit=None):
    """Join the new held-out subset to the immutable full-data zero-shot run."""
    with BASELINE_PREDICTIONS.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        image_path = str(Path(row["image_path"]).resolve())
        if image_path in lookup:
            raise RuntimeError("Original Node21 baseline contains a duplicate image")
        lookup[image_path] = row
    try:
        selected = [lookup[str(Path(path).resolve())] for path in spec["image_paths"]]
    except KeyError as error:
        raise RuntimeError("Held-out Node21 image is absent from the original baseline") from error
    if not np.array_equal([int(r["label:Nodule"]) for r in selected], spec["labels"].ravel()):
        raise RuntimeError("Held-out test labels differ from the original baseline")
    original = np.asarray([float(r["logit:Nodule"]) for r in selected]).reshape(-1, 1)
    np.testing.assert_allclose(cached_logits, original, atol=2e-5, rtol=1e-4)
    return {"passed": True, "images": len(selected),
            "comparison": "Same held-out images and labels joined by absolute image path",
            "source": str(BASELINE_PREDICTIONS), "source_sha256": sha256(BASELINE_PREDICTIONS),
            "max_logit_error": float(np.abs(cached_logits-original).max()),
            "atol": 2e-5, "rtol": 1e-4}


def train(args, specs, head, device, rank, world, cache_identity, run_identity):
    frozen_digest = tensor_digest(head.loss_module)
    counts = head.parameter_counts()
    if counts["trainable"] != 49952:
        raise RuntimeError("Unexpected trainable adapter parameter count")
    data = {name: TokenCache(args.cache_dir, name, spec["labels"]) for name, spec in specs.items()}
    sampler = DistributedSampler(data["train"], num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=False)
    train_loader = DataLoader(data["train"], sampler=sampler,
                              **loader_options(args, args.batch_size))
    wrapped = DistributedDataParallel(head, device_ids=[device.index], output_device=device.index,
                                      broadcast_buffers=False)
    optimizer = torch.optim.AdamW(head.trainable_parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    positives = int(specs["train"]["labels"].sum())
    negatives = len(data["train"]) - positives
    pos_weight = negatives / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    best_score, best_epoch, stale, initial_epoch = -math.inf, 0, 0, 1
    history = []
    last_path = args.output / "last.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device)
        if saved["run_fingerprint"] != run_identity["fingerprint"]:
            raise RuntimeError("Resume checkpoint configuration differs")
        head.adapter.load_state_dict(saved["adapter_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_score, best_epoch, stale = saved["best_score"], saved["best_epoch"], saved["stale"]
        initial_epoch = saved["epoch"] + 1
        history = saved["history"]
    baseline_valid_logits = np.load(args.cache_dir / "valid_baseline_logits.npy")
    baseline_valid = metrics(specs["valid"], baseline_valid_logits)
    if rank == 0:
        write_json(args.output / "baseline_validation_metrics.json", baseline_valid)
        save_predictions(args.output / "baseline_validation_predictions.csv",
                         specs["valid"], baseline_valid_logits)
        print(json.dumps({"stage": "training", "parameters": counts, "positive_weight": pos_weight,
                          "baseline_valid_auroc": baseline_valid["macro_auroc"],
                          "baseline_valid_auprc": baseline_valid["macro_auprc"]}), flush=True)
    for epoch in range(initial_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        timer = time.monotonic()
        sampler.set_epoch(epoch)
        head.train()
        loss_sum, examples = 0.0, 0
        first_grad_check = {}
        for tokens, labels, indices in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(tokens)["logits"]
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            for parameter in head.loss_module.parameters():
                if parameter.grad is not None:
                    raise RuntimeError("Frozen original scoring parameter received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(head.trainable_parameters(), 1.0, error_if_nonfinite=True)
            if not first_grad_check:
                first_grad_check = {"adapter_gradient_norm": float(norm), "frozen_gradients_none": True}
                if not float(norm) > 0:
                    raise RuntimeError("Adapter received no training gradient")
            optimizer.step()
            loss_sum += float(loss.detach()) * len(tokens)
            examples += len(tokens)
        aggregates = torch.tensor([loss_sum, examples], dtype=torch.float64, device=device)
        dist.all_reduce(aggregates)
        valid_logits = predict(head, data["valid"], args, device, rank, world)
        valid = metrics(specs["valid"], valid_logits)
        score = valid["macro_auroc"]
        if score is None:
            raise RuntimeError("Validation AUROC undefined")
        improved = score > best_score + 1e-4
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        entry = {
            "epoch": epoch, "train_weighted_bce": float(aggregates[0]/aggregates[1]),
            "train_examples_including_distributed_padding": int(aggregates[1]),
            "valid_auroc": score, "valid_auprc": valid["macro_auprc"],
            "learning_rate": optimizer.param_groups[0]["lr"], "selected": improved,
            "best_epoch": best_epoch, "best_valid_auroc": best_score,
            "elapsed_seconds": time.monotonic()-timer, "gradient_check": first_grad_check,
        }
        scheduler.step()
        history.append(entry)
        if tensor_digest(head.loss_module) != frozen_digest:
            raise RuntimeError("Frozen original scoring parameters changed")
        if rank == 0:
            if improved:
                atomic_save(args.output / "best_adapter.pt", {
                    "adapter_state_dict": {k:v.detach().cpu() for k,v in head.adapter.state_dict().items()},
                    "epoch": epoch, "validation_metrics": valid, "run_fingerprint": run_identity["fingerprint"],
                    "architecture": {"hidden_dim": 768, "bottleneck": 32, "scale": 0.1},
                    "prompt": PROMPT, "checkpoint": checkpoint_identity(),
                    "scope": "Supervised Node21 adaptation on a custom public-data split; original RadZero parameters frozen",
                })
                save_predictions(args.output / "best_validation_predictions.csv", specs["valid"], valid_logits)
                write_json(args.output / "best_validation_metrics.json", valid)
            atomic_save(last_path, {"adapter_state_dict": head.adapter.state_dict(),
                                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                   "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                                   "stale": stale, "history": history,
                                   "run_fingerprint": run_identity["fingerprint"]})
            write_json(args.output / "history.json", history)
            write_json(args.output / "progress.json", {
                "status": "training", "epoch": epoch, "max_epochs": args.epochs,
                "best_epoch": best_epoch, "best_valid_auroc": best_score,
                "stale_epochs": stale, "updated_at": now(), "last_epoch": entry,
            })
            print("EPOCH " + json.dumps(entry), flush=True)
        dist.barrier()
    del train_loader, wrapped, optimizer
    dist.barrier()
    best = torch.load(args.output / "best_adapter.pt", map_location=device)
    head.adapter.load_state_dict(best["adapter_state_dict"], strict=True)
    # The held-out test is evaluated only after validation-based model selection ends.
    test_logits = predict(head, data["test"], args, device, rank, world)
    test = metrics(specs["test"], test_logits)
    if rank == 0:
        baseline_test_logits = np.load(args.cache_dir / "test_baseline_logits.npy")
        reference = compare_original_test(specs["test"], baseline_test_logits, args.limit_test)
        baseline_test = metrics(specs["test"], baseline_test_logits)
        save_predictions(args.output / "test_predictions.csv", specs["test"], test_logits)
        save_predictions(args.output / "baseline_test_predictions.csv",
                         specs["test"], baseline_test_logits)
        write_json(args.output / "test_metrics.json", test)
        write_json(args.output / "baseline_test_metrics.json", baseline_test)
        write_json(args.output / "test_baseline_equivalence.json", reference)
        rows = []
        for split, baseline, adapted in (
            ("valid", baseline_valid, best["validation_metrics"]), ("test", baseline_test, test)
        ):
            row = {"split": split, "images": len(specs[split]["image_paths"])}
            for metric_name in ("auroc", "auprc"):
                b, a = baseline["macro_"+metric_name], adapted["macro_"+metric_name]
                row["baseline_"+metric_name] = b
                row["adapter_"+metric_name] = a
                row["delta_"+metric_name+"_percentage_points"] = (a-b)*100
            rows.append(row)
        with (args.output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        status = "smoke_complete" if any(getattr(args, "limit_"+s) for s in SPLITS) else "complete"
        summary = {
            "status": status, "completed_at": now(), "best_epoch": best_epoch,
            "epochs_completed": len(history), "early_stopped": len(history) < args.epochs,
            "parameters": counts, "frozen_scoring_sha256_before": frozen_digest,
            "frozen_scoring_sha256_after": tensor_digest(head.loss_module),
            "original_model_frozen": True, "counts": {s:len(x["image_paths"]) for s,x in specs.items()},
            "class_counts": {s: {"positive": int(x["labels"].sum()),
                                  "negative": int(len(x["labels"]) - x["labels"].sum())}
                             for s, x in specs.items()},
            "selection": "Highest validation AUROC among trained epochs, min_delta 0.0001; "
                         "test never used for model selection; no guarantee of improvement over epoch-zero baseline.",
            "best_checkpoint": str(args.output / "best_adapter.pt"), "comparison": rows,
            "baseline_test_equivalence": reference, "run_fingerprint": run_identity["fingerprint"],
            "cache_fingerprint": cache_identity["fingerprint"], "history": history,
            "validation": best["validation_metrics"], "test": test,
            "split_manifest": str(args.splits_dir / "split_manifest.json"),
            "split_protocol": specs["test"].get("metadata", {}),
            "scope": "Image-level nodule classification with Node21 supervision on a custom held-out split; "
                     "not zero-shot, not an official Node21 challenge test, and not detection/FROC evaluation.",
        }
        write_json(args.output / "summary.json", summary)
        write_json(args.output / "progress.json", {
            "status": status, "best_epoch": best_epoch, "epochs_completed": len(history),
            "completed_at": summary["completed_at"], "comparison": rows,
        })
        print("TRAINING COMPLETE " + json.dumps(rows), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS_DIR/"outputs/node21_adapter_gpu01")
    parser.add_argument("--splits-dir", type=Path, default=ASSETS_DIR/"splits/node21_adapter_v1")
    parser.add_argument("--cache-dir", type=Path, default=ASSETS_DIR/"features/node21_adapter_fp32_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    for name in SPLITS:
        parser.add_argument("--limit-"+name, type=int, default=0)
    args = parser.parse_args()
    for key in ("output", "splits_dir", "cache_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    outputs_root, features_root = (ASSETS_DIR/"outputs").resolve(), (ASSETS_DIR/"features").resolve()
    if (args.output.parent != outputs_root or not args.output.name.startswith("node21_adapter_")
            or args.cache_dir.parent != features_root
            or not args.cache_dir.name.startswith("node21_adapter_")):
        parser.error("Use independent node21_adapter_* directories directly beneath assets/outputs "
                     "and assets/features; existing baseline and SIIM outputs are protected")
    if (min(args.epochs, args.patience, args.batch_size, args.extract_batch_size) < 1
            or args.workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0
            or any(getattr(args,"limit_"+s)<0 for s in SPLITS)):
        parser.error("Invalid training hyperparameters")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        parser.error("This run is restricted to physical GPU 0 and GPU 1: CUDA_VISIBLE_DEVICES=0,1")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("Launch with torchrun --nproc_per_node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        specs = select_specs(args)
        identity = {
            "version": "frozen-radzero-node21-residual-adapter-v1", "started_at": now(),
            "hyperparameters": {k:v for k,v in vars(args).items() if not isinstance(v, Path)},
            "paths": {k:str(getattr(args,k)) for k in ("output","splits_dir","cache_dir")},
            "architecture": {"hidden_dim":768,"bottleneck":32,"scale":0.1,"up_zero_initialized":True},
            "prompt": PROMPT, "precision":"float32; TF32 disabled",
            "physical_gpus":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size":world, "checkpoint":checkpoint_identity(),
            "split_manifest_sha256":sha256(args.splits_dir/"split_manifest.json"),
            "split_csv_sha256": {name: sha256(args.splits_dir / (name + ".csv")) for name in SPLITS},
            "code_sha256": source_hashes(),
            "model_source_sha256": model_source_hashes(),
            "model_config_sha256": sha256(MODEL_DIR / "config.json"),
            "preprocessor_sha256": sha256(MODEL_DIR / "preprocessor_config.json"),
            "tokenizer_config_sha256": sha256(MODEL_DIR / "tokenizer_config.json"),
            "tokenizer_sha256": tokenizer_hashes(),
            "baseline_source": str(BASELINE_PREDICTIONS),
            "baseline_source_sha256": sha256(BASELINE_PREDICTIONS),
            "baseline_run_manifest_sha256": sha256(BASELINE_DIRECTORY / "run_manifest.json"),
            "scope": "Custom supervised Node21 split of the public 4882-image training release; "
                     "unknown patient identities prevent a full patient-disjointness guarantee.",
        }
        # Creation time is not part of the identity used for safe resumption.
        identity["fingerprint"] = fingerprint({k:v for k,v in identity.items() if k!="started_at"})
        if rank == 0:
            manifest_path = args.output/"run_manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text())["fingerprint"] != identity["fingerprint"]:
                    raise RuntimeError("Output belongs to another training configuration")
            else:
                write_json(manifest_path,identity)
            write_json(args.output/"progress.json",{"status":"extracting_features","started_at":now()})
        dist.barrier()
        cache_identity = prepare_cache(args,specs,rank,world)
        model,processor,tokenizer = load_model(device)
        model.requires_grad_(False)
        head = build_head(model,tokenizer,prompt=PROMPT,bottleneck=32,scale=0.1).to(device)
        extract_features(args,specs,model,processor,head,device,rank,world,cache_identity)
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError("Original model was not entirely frozen during extraction")
        del model,processor,tokenizer
        torch.cuda.empty_cache()
        train(args,specs,head,device,rank,world,cache_identity,identity)
        write_json(args.output/("rank"+str(rank))/"completion.json",{
            "status":"complete","rank":rank,"device":str(device),
            "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name":torch.cuda.get_device_name(local_rank),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "completed_at":now(),
        })
    except BaseException as error:
        write_json(args.output/("rank"+str(rank)+"_error.json"),{
            "status":"failed","error":repr(error),"time":now(),"traceback":traceback.format_exc()})
        if rank==0:
            write_json(args.output/"progress.json",{"status":"failed","error":repr(error),"failed_at":now()})
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
