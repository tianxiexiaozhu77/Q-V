#!/usr/bin/env python3
"""Evaluate one frozen released AlphaRAD checkpoint with audited local manifests."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "local_assets"
os.environ.setdefault("ALPHARAD_ASSETS", str(ASSETS))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ASSETS / ".cache/matplotlib"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    p.add_argument("--check-data-only", action="store_true")
    args = p.parse_args()
    start = time.time()
    entry = json.loads(args.catalog.read_text())["datasets"][args.dataset]
    csv_path = Path(entry["csv_path"])
    class_names = entry["class_names"]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.json").exists():
        raise FileExistsError("Completed result already exists; use a fresh output directory")

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
    from cxrclip import seed_everything
    from cxrclip.evaluator_utils import disease_list_mapping, disease_name_mapping
    from cxrclip.prompt import constants

    # Public code omits constants for several benchmarks; local class ordering
    # is explicit in the manifest, never inferred from model predictions.
    if hasattr(constants, args.dataset.upper()):
        if list(getattr(constants, args.dataset.upper())) != class_names:
            raise ValueError("Manifest class order disagrees with author's constants")
    else:
        setattr(constants, args.dataset.upper(), class_names)
    csv_sha = sha256(csv_path)
    if csv_sha != entry["csv_sha256"]:
        raise ValueError("Frozen manifest CSV hash changed")
    df = pd.read_csv(csv_path)
    if len(df) != entry["actual_count"]:
        raise ValueError("Frozen manifest row count changed")
    expected_labels = []
    paths = []
    for row in df.to_dict("records"):
        path = str(row["image"])
        if path.startswith("["):
            path = ast.literal_eval(path)[0]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        if entry["task"] == "classification":
            names = ast.literal_eval(row["class"]) if row["class"].startswith("[") else [row["class"]]
            y = [int(disease_name_mapping(c) in disease_list_mapping(names)) for c in class_names]
            original = ast.literal_eval(row["label"]) if isinstance(row["label"], str) else [row["label"]]
            if y != original:
                raise ValueError(f"Label/class disagreement for {path}: {y} != {original}")
            expected_labels.append(y)
    if len(set(paths)) != entry.get("unique_images", len(set(paths))):
        raise ValueError("Frozen unique-image count changed")
    if expected_labels and "positive_count" in entry:
        if np.asarray(expected_labels).sum(axis=0).tolist() != entry["positive_count"]:
            raise ValueError("Frozen per-class positive count changed")
    audit = {"dataset": args.dataset, "manifest": entry,
             "csv_sha256": csv_sha, "catalog_sha256": sha256(args.catalog),
             "rows": len(df), "unique_images": len(set(paths)),
             "paths_present": True, "label_class_mapping_checked": entry["task"] == "classification"}
    write_json(args.output / "data_audit.json", audit)
    if args.check_data_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("No assigned CUDA GPU available")
    torch.set_num_threads(args.threads)
    seed_everything(1234)
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9":
        raise ValueError("Checkpoint SHA256 differs from verified official release")
    loader = {"batch_size": args.batch_size, "num_workers": args.workers,
              "pin_memory": True, "shuffle": False, "drop_last": False}
    if args.workers:
        loader["prefetch_factor"] = 2
    config = {
        "base": {"seed": 1234},
        "test": {"seed": 1234, "checkpoint": [str(args.checkpoint)]},
        "data_test": {args.dataset: {"name": args.dataset,
                       "data_type": "image_classification", "data_path": str(csv_path),
                       "n_class": len(class_names), "normalize": None}},
        "dataloader": {"test": loader},
        "seg_save_dir": str(args.output / "segmentation"),
        "prediction_save_dir": str(args.output / "predictions"),
    }
    write_json(args.output / "eval_config.json", config)
    from cxrclip.evaluator import Evaluator
    evaluator = Evaluator(config, [str(args.checkpoint)], "test")
    with torch.inference_mode():
        author_metrics = evaluator.evaluate_clip_offline(str(args.checkpoint), args.dataset)
    result = {**audit, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
              "training": False, "adaptation": False, "author_metrics": author_metrics,
              "batch_size": args.batch_size, "seed": 1234,
              "device": torch.cuda.get_device_name(0),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__}
    if entry["task"] == "classification":
        prediction_dir = args.output / "predictions" / args.checkpoint.stem / args.dataset
        scores = np.load(prediction_dir / "predictions.npy").reshape(len(df), len(class_names))
        labels = np.load(prediction_dir / "labels.npy")
        if not np.array_equal(labels, np.asarray(expected_labels)):
            raise ValueError("Saved evaluator labels differ from audited CSV")
        saved_paths = json.loads((prediction_dir / "image_paths.json").read_text())
        if saved_paths != df["image"].tolist():
            raise ValueError("Saved predictions changed manifest order")
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite model prediction")
        metrics = {}
        for i, name in enumerate(class_names):
            y, score = labels[:, i], scores[:, i]
            if len(np.unique(y)) != 2:
                metrics[name] = {"positive_count": int(y.sum()), "auroc": None,
                                 "average_precision": None, "auprc_trapezoidal": None}
                continue
            precision, recall, _ = precision_recall_curve(y, score)
            metrics[name] = {"positive_count": int(y.sum()),
                             "auroc": float(roc_auc_score(y, score)),
                             "average_precision": float(average_precision_score(y, score)),
                             "auprc_trapezoidal": float(auc(recall, precision))}
        valid = [v for v in metrics.values() if v["auroc"] is not None]
        result["classification"] = {"per_class": metrics, "valid_class_count": len(valid),
            **{"macro_" + k: float(np.mean([v[k] for v in valid])) if valid else None
               for k in ("auroc", "average_precision", "auprc_trapezoidal")}}
        # Independent recomputation must match the original evaluator.
        upstream = author_metrics["zeroshot_binary"]
        if len(valid) == len(class_names):
            if not np.isclose(result["classification"]["macro_auroc"], upstream["AUROC(Avg)"], rtol=0, atol=1e-10):
                raise ValueError("Independent AUROC differs from author metric")
            if not np.isclose(result["classification"]["macro_average_precision"], upstream["PR_AUROC(Avg)"], rtol=0, atol=1e-10):
                raise ValueError("Independent AP differs from author metric")
    result["elapsed_seconds"] = time.time() - start
    result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    write_json(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
