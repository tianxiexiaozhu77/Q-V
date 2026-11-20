#!/usr/bin/env python3
"""Independent saved-prediction audit; no models, training, or GPU operations.

Classification uses rank statistics and cumulative confusion counts instead of
the evaluation worker's sklearn metrics. Spatial verification independently
reconstructs native map peak coordinates and counts box hits, without importing
the author's pointing-game evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.stats import rankdata


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(got, expected, context):
    if not np.isclose(got, expected, atol=1e-12, rtol=0):
        raise AssertionError(f"{context}: computed={got} reported={expected}")
    return float(abs(got - expected))


def binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if not npos or not nneg:
        return {key: None for key in ("auroc", "auprc", "average_precision")}
    ranks = rankdata(scores, method="average")
    auroc = (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(labels) - 1]
    tp = np.cumsum(sorted_labels)[ends]
    precision = tp / (ends + 1)
    recall = tp / npos
    ap = np.sum(np.diff(np.r_[0., recall]) * precision)
    auprc = np.trapz(np.r_[1., precision], np.r_[0., recall])
    return {"auroc": float(auroc), "auprc": float(auprc), "average_precision": float(ap)}


def spatial_audit(cache, protocol, reported):
    import torch
    from torch.nn import functional as F
    from PIL import Image

    torch.set_num_threads(4)
    spec = protocol["spatial_manifest"]
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("Spatial labels no longer match frozen protocol")
    reference = spec["reference"]
    if sha(reference["result"]) != reference["sha256"]:
        raise ValueError("Previous spatial reference result changed")
    rows = json.loads(Path(spec["path"]).read_text())["rows"]
    classes = protocol["class_names"]
    branches = list(reported)
    with np.load(cache / "native_maps.npz", allow_pickle=False) as archive:
        paths = archive["image_paths"].tolist()
        if archive["class_names"].tolist() != classes:
            raise ValueError("Map class order differs")
        lookup = {path: i for i, path in enumerate(paths)}
        maps = np.stack([archive[key] for key in branches])
    if maps.shape[2:] != (len(classes), 1370):
        raise ValueError("Expected published AlphaRAD CLS+37x37 map")
    hits = np.zeros((len(branches), len(classes)), dtype=np.int64)
    counts = np.zeros(len(classes), dtype=np.int64)
    evaluated_rows = 0
    with torch.inference_mode():
        for row in rows:
            if row["image"] not in lookup:
                raise ValueError("Missing spatial test image")
            index = lookup[row["image"]]
            annotated = []
            all_boxes = []
            for c, name in enumerate(classes):
                boxes = [box for box, label in zip(row["boxes"], row["class_names_present"]) if label == name]
                if boxes:
                    annotated.append(c)
                    all_boxes.append(boxes)
            if not annotated:
                continue
            with Image.open(row["image"]) as image:
                width, height = image.size
            values = maps[:, index][:, annotated, 1:].reshape(len(branches) * len(annotated), 1, 37, 37)
            output_hw = (height, width) if height == width else (518, 518)
            resized = F.interpolate(torch.from_numpy(values.copy()), size=output_hw,
                                    mode="bilinear", align_corners=False)
            peak = resized.flatten(1).argmax(1).numpy().reshape(len(branches), len(annotated))
            for a, (c, boxes) in enumerate(zip(annotated, all_boxes)):
                counts[c] += 1
                for m in range(len(branches)):
                    yy, xx = divmod(int(peak[m, a]), output_hw[1])
                    if height != width:
                        scale = 518 / min(height, width)
                        resized_h, resized_w = int(round(height * scale)), int(round(width * scale))
                        top, left = max((resized_h - 518) // 2, 0), max((resized_w - 518) // 2, 0)
                        xx, yy = int(round((xx + left) / scale)), int(round((yy + top) / scale))
                    hit = any(float(b[0]) <= xx <= float(b[2]) and float(b[1]) <= yy <= float(b[3]) for b in boxes)
                    hits[m, c] += int(hit)
            evaluated_rows += 1
    result = {}
    for m, key in enumerate(branches):
        per_class, max_error = {}, 0.
        for c, name in enumerate(classes):
            if not counts[c]:
                continue
            pg = float(hits[m, c] / counts[c])
            error = check(pg, reported[key]["per_class"][name]["pointing_game"], f"{key}/{name}/PG")
            max_error = max(max_error, error)
            per_class[name] = {"hits": int(hits[m, c]), "annotated_image_class_pairs": int(counts[c]), "pointing_game": pg}
        macro = float(np.mean([v["pointing_game"] for v in per_class.values()]))
        check(macro, reported[key]["pointing_game"], f"{key}/macro PG")
        check(reported[key]["delta_pp"], 100 * (macro - reported["original"]["pointing_game"]), f"{key}/PG delta")
        if reported[key]["rows"] != len(rows):
            raise ValueError("Spatial row count changed")
        result[key] = {"rows": len(rows), "annotated_rows": evaluated_rows,
            "pointing_game": macro, "per_class": per_class, "max_reported_error": max_error}
    return result


def audit(evaluation, protocol_path):
    start = time.time()
    evaluation, protocol_path = Path(evaluation), Path(protocol_path)
    record = json.loads((evaluation / "result.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Independent full audit requires a completed full run")
    protocol = json.loads(protocol_path.read_text())
    if sha(protocol_path) != record["protocol_sha256"]:
        raise ValueError("Protocol changed")
    spec = protocol["test_manifest"]
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("Frozen classification labels changed")
    test = json.loads(Path(spec["path"]).read_text())["rows"]
    labels = np.asarray([row["labels"] for row in test], dtype=np.int64)
    paths = [row["image"] for row in test]
    if len(test) != record["test_count"] or paths != protocol["test_images"]:
        raise ValueError("Test identities/count changed")
    classes = protocol["class_names"]
    if sha(evaluation / "predictions.npz") != record["predictions_sha256"]:
        raise ValueError("Saved predictions changed")
    classification = {}
    max_error = 0.
    with np.load(evaluation / "predictions.npz", allow_pickle=False) as predictions:
        if predictions["image_paths"].tolist() != paths or predictions["class_names"].tolist() != classes:
            raise ValueError("Prediction identities/order changed")
        for name, method in record["methods"].items():
            scores = predictions[name]
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError("Bad score shape or values")
            per_class = {}
            for c, cls in enumerate(classes):
                computed = binary_metrics(labels[:, c], scores[:, c])
                for key, value in computed.items():
                    expected = method["metrics"]["per_class"][cls][key]
                    if value is None:
                        if expected is not None:
                            raise ValueError("Undefined class metric discrepancy")
                    else:
                        max_error = max(max_error, check(value, expected, f"{name}/{cls}/{key}"))
                per_class[cls] = computed
            macros = {"macro_" + key: float(np.mean([v[key] for v in per_class.values() if v[key] is not None]))
                      for key in ("auroc", "auprc", "average_precision")}
            for key, value in macros.items():
                max_error = max(max_error, check(value, method["metrics"][key], f"{name}/{key}"))
            for key, expected in method["delta_pp"].items():
                check(100 * (macros[key] - record["methods"]["original"]["metrics"][key]), expected,
                      f"{name}/{key}/delta")
            classification[name] = {"per_class": per_class, **macros}
    spatial = spatial_audit(Path(record["cache"]), protocol, record["spatial"]) if record["spatial"] else {}
    output = {"status": "passed", "model": record["model"], "dataset": record["dataset"],
        "classification_rows": len(test), "classification_methods": len(classification),
        "classification_algorithm": "Wilcoxon rank-sum AUROC with averaged ties; sorted-score grouped cumulative TP/FP trapezoidal PR area and AP; independent of evaluate.py and sklearn metrics",
        "spatial_algorithm": "Independently remove CLS; bilinear align_corners=False native-map resize; argmax; inverse native resize/crop coordinates; inclusive GT-box intersection; disease macro PG",
        "classification_max_abs_difference": max_error, "absolute_tolerance": 1e-12,
        "classification": classification, "spatial": spatial, "gpu_used": False,
        "test_labels_used_only_for_metrics": True, "elapsed_seconds": time.time() - start,
        "source_sha256": {str(path): sha(path) for path in [Path(__file__).resolve(), protocol_path.resolve(),
            evaluation / "result.json", evaluation / "predictions.npz", Path(spec["path"])]}}
    if spatial:
        output["source_sha256"][str(Path(record["cache"]) / "native_maps.npz")] = sha(Path(record["cache"]) / "native_maps.npz")
        output["source_sha256"][protocol["spatial_manifest"]["path"]] = sha(protocol["spatial_manifest"]["path"])
    target = evaluation / "independent_metric_audit.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps({k: output[k] for k in ["status", "model", "dataset", "classification_methods", "classification_max_abs_difference", "elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    audit(args.evaluation, args.protocol)
