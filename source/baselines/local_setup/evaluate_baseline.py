#!/usr/bin/env python3
"""Run a frozen pretrained adapter on the exact phase6 evaluation manifests.

Each model keeps its own image preprocessing and scoring. No support images,
optimizer, fine tuning, score calibration, or test-driven prompt selection.
Unsupported class/task combinations are explicit, never reduced-class macros.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    import numpy as np
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=jsonable) + "\n")
    temp.replace(path)


def classification_metrics(labels, scores, classes):
    import numpy as np
    from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
    per_class = {}
    for i, name in enumerate(classes):
        y, p = labels[:, i], scores[:, i]
        if len(np.unique(y)) != 2:
            per_class[name] = {"positive_count": int(y.sum()), "auroc": None, "auprc": None, "average_precision": None}
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        per_class[name] = {"positive_count": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(auc(recall, precision)), "average_precision": float(average_precision_score(y, p))}
    valid = [row for row in per_class.values() if row["auroc"] is not None]
    return {"per_class": per_class, "valid_class_count": len(valid), "class_count": len(classes),
        **{"macro_" + key: float(np.mean([row[key] for row in valid])) if valid else None
           for key in ("auroc", "auprc", "average_precision")}}


def decode_rle(value, width, height):
    """Exact relative-gap Fortran RLE convention in the frozen AlphaRAD CSV."""
    import numpy as np
    output = np.zeros(width * height, dtype=bool)
    runs = [int(x) for x in value.split()]
    if len(runs) % 2:
        raise ValueError("Odd RLE count")
    cursor = 0
    for gap, count in zip(runs[::2], runs[1::2]):
        cursor += gap
        if gap < 0 or count < 0 or cursor + count > len(output):
            raise ValueError("Out-of-range RLE")
        output[cursor:cursor + count] = True
        cursor += count
    return output.reshape(width, height).T


def spatial_batch(maps, rows, classes, task):
    import numpy as np
    from PIL import Image
    if isinstance(maps, np.ndarray):
        maps = list(maps)
    if not isinstance(maps, (tuple, list)) or len(maps) != len(rows):
        raise ValueError("grounding must return one [C,H_original,W_original] array per image")
    result = []
    thresholds = np.arange(0, 1.01, .01)
    for values, row in zip(maps, rows):
        values = np.asarray(values)
        with Image.open(row["image"]) as image:
            width, height = image.size
        if values.shape != (len(classes), height, width):
            raise ValueError(f"Map is not in original image coordinates: {values.shape}, {(len(classes), height, width)}")
        if np.isinf(values).any():
            raise ValueError("Infinite heatmap; use NaN only for uncovered crop pixels")
        items = []
        for c, name in enumerate(classes):
            raw = values[c]
            valid = np.isfinite(raw)
            if not valid.any():
                raise ValueError(f"Empty spatial coverage: {name}")
            matching = [i for i, label in enumerate(row["class_names_present"]) if label == name]
            if not matching:
                items.append(None)
                continue
            if task == "grounding":
                y, x = np.unravel_index(np.argmax(np.where(valid, raw, -np.inf)), raw.shape)
                boxes = [row["boxes"][i] for i in matching]
                hit = any(float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3]) for box in boxes)
                items.append({"hit": int(hit), "peak_xy": [int(x), int(y)], "valid_fraction": float(valid.mean())})
            else:
                # Retain the existing positive-only macro Dice definition and
                # 0:.01:1 threshold sweep. Crop-excluded pixels are background.
                gt = np.zeros((height, width), dtype=bool)
                for i in matching:
                    gt |= decode_rle(row["encoded_pixels"][i], width, height)
                if not gt.any():
                    raise ValueError("Positive segmentation row has no foreground")
                low, high = float(raw[valid].min()), float(raw[valid].max())
                normalized = (raw[valid] - low) / (high - low + 1e-8)
                prob = np.zeros((height, width), dtype=np.float32)
                prob[valid] = 1 / (1 + np.exp(-normalized))
                all_scores = np.sort(prob.ravel())
                foreground_scores = np.sort(prob[gt])
                predicted = len(all_scores) - np.searchsorted(all_scores, thresholds, side="right")
                true_positive = len(foreground_scores) - np.searchsorted(foreground_scores, thresholds, side="right")
                dice = 2 * true_positive / (predicted + int(gt.sum()))
                items.append({"dice_by_threshold": dice.tolist(), "positive_pixels": int(gt.sum()),
                    "valid_fraction": float(valid.mean()), "raw_minmax": [low, high]})
        result.append(items)
    return result


def spatial_metrics(items, classes, task):
    import numpy as np
    per_class = {}
    for c, name in enumerate(classes):
        valid = [row[c] for row in items if row[c] is not None]
        if not valid: continue
        if task == "grounding":
            per_class[name] = {"count": len(valid), "pointing_game": float(np.mean([v["hit"] for v in valid]))}
        else:
            curve = np.asarray([v["dice_by_threshold"] for v in valid]).mean(0)
            best = int(curve.argmax())
            per_class[name] = {"count": len(valid), "dice": float(curve[best]), "best_threshold": float(np.arange(0, 1.01, .01)[best]), "mean_dice_curve": curve.tolist()}
    key = "pointing_game" if task == "grounding" else "dice"
    return {key: float(np.mean([row[key] for row in per_class.values()])) if per_class else None,
        "per_class": per_class, "valid_class_count": len(per_class),
        "protocol": "Native original-coordinate raw maps; NaN crop coverage ignored for PG. Positive-only samplewise Dice: per-map minmax, sigmoid, uncovered pixels=0, one test-wide threshold per class in 0:.01:1. Threshold search uses test masks, matching the exploratory phase6 setting."}


def load_adapter(root, device):
    sys.path[:0] = [str(root / "local_setup"), str(root)]
    path = root / "local_setup/adapter.py"
    spec = importlib.util.spec_from_file_location("baseline_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_adapter(device=device)


def run(args):
    import numpy as np
    import torch
    start = time.time()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU")
    catalog_sha = sha(args.catalog)
    catalog = json.loads(args.catalog.read_text())
    dataset_names = args.datasets or list(catalog["datasets"])
    sources = {str(Path(__file__).resolve()): sha(__file__),
        str(args.model_root / "local_setup/adapter.py"): sha(args.model_root / "local_setup/adapter.py")}
    model_manifest = args.model_root / "local_setup/manifest.json"
    if model_manifest.exists():
        sources[str(model_manifest)] = sha(model_manifest)
    write(args.output / "source_sha256.json", sources)
    write(args.output / "status.json", {"status": "loading", "model": args.model_root.name, "started_at": now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "catalog_sha256": catalog_sha})
    adapter = load_adapter(args.model_root, args.device)
    results = {}
    for key in dataset_names:
        entry = catalog["datasets"][key]
        manifest_path = Path(entry["manifest"])
        if sha(manifest_path) != entry["sha256"]:
            raise ValueError("Changed fixed evaluation manifest: " + key)
        data = json.loads(manifest_path.read_text())
        output = args.output / key
        output.mkdir(parents=True, exist_ok=False)
        classes, task = data["class_names"], data["task"]
        rows = data["rows"][:args.smoke] if args.smoke else data["rows"]
        began = time.time()
        try:
            if hasattr(adapter, "set_dataset"):
                adapter.set_dataset(data)
            if hasattr(adapter, "check_classes"):
                adapter.check_classes(classes)
            if task != "classification" and not callable(getattr(adapter, "grounding", None)):
                raise NotImplementedError("No verified native grounding implementation")
            write(output / "manifest.json", {k: v for k, v in data.items() if k != "rows"})
            arrays, spatial_items = [], []
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                paths = [r["image"] for r in batch]
                with torch.inference_mode():
                    if task == "classification":
                        values = adapter.predict(paths, classes)
                        if isinstance(values, torch.Tensor): values = values.detach().cpu().numpy()
                        values = np.asarray(values)
                        if values.shape != (len(batch), len(classes)) or not np.isfinite(values).all():
                            raise ValueError(f"Invalid score shape or values: {values.shape}")
                        arrays.append(values)
                    else:
                        maps = adapter.grounding(paths, classes)
                        spatial_items.extend(spatial_batch(maps, batch, classes, task))
                        if offset == 0:
                            np.savez_compressed(output / "map_examples.npz", **{f"image_{i}": np.asarray(v) for i, v in enumerate(maps)})
                done = offset + len(batch)
                progress = {"status": "inference", "model": args.model_root.name, "dataset": key,
                    "task": task, "images_completed": done, "images_total": len(rows),
                    "elapsed_seconds": time.time() - began, "updated_at": now()}
                write(args.output / "progress.json", progress)
                if offset == 0 or done == len(rows) or done % (args.batch_size * 50) == 0:
                    print(json.dumps(progress), flush=True)
            if task == "classification":
                scores = np.concatenate(arrays)
                labels = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
                np.savez_compressed(output / "predictions.npz", scores=scores, labels=labels,
                    image_paths=np.asarray([r["image"] for r in rows]), class_names=np.asarray(classes))
                metrics = classification_metrics(labels, scores, classes)
            else:
                write(output / "spatial_predictions.json", spatial_items)
                metrics = spatial_metrics(spatial_items, classes, task)
            result = {"status": "smoke_passed" if args.smoke else "complete", "model": args.model_root.name,
                "dataset": key, "task": task, "rows": len(rows), "expected_rows": data["count"],
                "class_names": classes, "metrics": metrics, "manifest_sha256": entry["sha256"],
                "comparison_references": data["comparison_references"],
                "adapter_metadata": getattr(adapter, "metadata", {}), "support_examples_used": 0,
                "network_weight_updates": False, "elapsed_seconds": time.time() - began,
                "score_handling": "native adapter scores; no extra sigmoid or cross-class softmax"}
        except NotImplementedError as error:
            result = {"status": "unsupported", "model": args.model_root.name, "dataset": key,
                "task": task, "expected_rows": data["count"], "class_names": classes,
                "reason": str(error), "adapter_metadata": getattr(adapter, "metadata", {}),
                "note": "No classes or images silently dropped; not a comparable full-task result."}
        except Exception as error:
            result = {"status": "failed", "model": args.model_root.name, "dataset": key,
                "task": task, "error": repr(error), "traceback": traceback.format_exc()}
            if args.device == "cuda": torch.cuda.empty_cache()
        write(output / "result.json", result)
        results[key] = result
        print(json.dumps({"dataset": key, "status": result["status"], "metrics": result.get("metrics", {})}, default=jsonable), flush=True)
    changed = [path for path, digest in sources.items() if sha(path) != digest]
    if changed or sha(args.catalog) != catalog_sha:
        raise ValueError("Sources/catalog changed during evaluation: " + str(changed))
    failed = [key for key, result in results.items() if result["status"] == "failed"]
    status = "failed" if failed else "smoke_passed" if args.smoke else "complete"
    write(args.output / "status.json", {"status": status, "model": args.model_root.name,
        "started_at_epoch": start, "finished_at": now(), "elapsed_seconds": time.time() - start,
        "jobs": {key: value["status"] for key, value in results.items()}, "failed": failed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "source_hashes_unchanged": True,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated()/1024**2 if args.device == "cuda" else 0})
    if failed:
        raise RuntimeError("Failed tasks: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.smoke < 0 or args.threads < 1:
        parser.error("Invalid batch/threads/smoke")
    args.model_root = args.model_root.resolve()
    args.catalog = args.catalog.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
