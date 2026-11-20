"""Frozen AlphaRAD QaTa Dice semantics; no threshold selection on test data.

The author first min/max normalizes all native tokens, INCLUDING CLS, removes
CLS, bilinearly upsamples to the square image size, then applies sigmoid.
Predictions use strict `probability > threshold` in float32. Dice is foreground,
positive-only, samplewise, with every image receiving equal weight.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
DISPLAY = dict(zip(BRANCHES, ("Original", "Q", "V", "Q+V")))
THRESHOLDS = np.arange(0, 1.01, 0.01)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with p.open("x", encoding="utf-8") as f:
            f.write(encoded)
            f.flush()
            import os
            os.fsync(f.fileno())
    else:
        temporary = p.with_name(p.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(p)


def read_protocol(path: str | Path) -> tuple[Path, dict]:
    p = Path(path).resolve()
    document = json.loads(p.read_text())
    # The canonical protocol has split descriptors, never embedded test masks.
    for field in ("protocol_path", "protocol_file"):
        if field in document:
            q = Path(document[field])
            p = q if q.is_absolute() else p.parent / q
            document = json.loads(p.read_text())
            break
    if (document.get("status") != "frozen" or document.get("eta") != .3 or
            document.get("crop_fraction") != .8 or document.get("class_names") != ["covid"] or
            not document.get("evaluation_positive_only")):
        raise ValueError("Expected frozen eta=.3 center80 positive-only QaTa protocol")
    if not np.array_equal(np.asarray(document["threshold_protocol"]["grid"]), THRESHOLDS):
        raise ValueError("Protocol threshold grid differs from the original 0:.01:1 rule")
    return p, document


def split_descriptor(protocol_path: Path, protocol: dict, split: str) -> tuple[Path, str | None]:
    descriptor = protocol.get(f"{split}_rows", protocol.get(f"{split}_rows_file"))
    if descriptor is None:
        descriptor = protocol.get("splits", {}).get(split)
    if descriptor is None:
        descriptor = protocol.get(split)
    if isinstance(descriptor, list):
        raise ValueError("Use a protocol with independent split files, not embedded test annotations")
    if isinstance(descriptor, dict):
        name = descriptor.get("path", descriptor.get("file", descriptor.get("rows_path")))
        digest = descriptor.get("sha256", descriptor.get("rows_sha256"))
    elif isinstance(descriptor, str):
        name, digest = descriptor, protocol.get(f"{split}_rows_sha256")
    else:
        name, digest = f"{split}_rows.json", None
    p = Path(name)
    if not p.is_absolute():
        p = protocol_path.parent / p
    return p.resolve(), digest


def load_rows(path: Path, expected_sha: str | None = None) -> list[dict]:
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError(f"Split manifest SHA256 mismatch: {path}")
    obj = json.loads(path.read_text())
    rows = obj if isinstance(obj, list) else obj["rows"]
    if not rows:
        raise ValueError("Empty evaluation split")
    images = [row.get("image", row.get("image_path")) for row in rows]
    if len(set(images)) != len(images):
        raise ValueError("Duplicate image rows in evaluation split")
    return rows


def row_image(row: dict) -> str:
    return row.get("image", row.get("image_path"))


def row_patient(row: dict) -> str:
    p = row.get("patient_id")
    if not p or not p.startswith("sub-S"):
        raise ValueError("QaTa patient bootstrap requires explicit BIMCV sub-S subject IDs")
    return p


def decode_relative_rle(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    """Verbatim indexing convention of AlphaRAD.evaluator_utils.rle2mask."""
    values = np.asarray([int(x) for x in encoded.split()], dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE must have alternating relative starts and lengths")
    flat = np.zeros(math.prod(shape), dtype=bool)
    position = 0
    for start, length in values.reshape(-1, 2):
        if start < 0 or length < 0:
            raise ValueError("Invalid negative RLE position/length")
        position += int(start)
        if position + int(length) > len(flat):
            raise ValueError("RLE extends outside image")
        flat[position:position + int(length)] = True
        position += int(length)
    return flat.reshape(shape[1], shape[0]).T


def load_positive_masks(rows: list[dict]) -> tuple[np.ndarray, dict]:
    masks, audit = [], {"count": len(rows), "positive_only": True, "source_files": []}
    for row in rows:
        image_path = row_image(row)
        with Image.open(image_path) as im:
            shape = (im.height, im.width)
        if shape[0] != shape[1]:
            raise ValueError("This frozen QaTa evaluator requires square source images")
        names = row.get("class_names_present", row.get("class", "covid"))
        if isinstance(names, str):
            names = [names]
        if names != ["covid"]:
            raise ValueError("Expected one annotated covid class in each positive image")
        encoded = row.get("encoded_pixels", row.get("EncodedPixels"))
        if isinstance(encoded, str):
            encoded = ast.literal_eval(encoded) if encoded.startswith("[") else [encoded]
        truth = None
        if encoded is not None:
            if not encoded or encoded[0].strip() == "-1":
                raise ValueError("Negative/empty masks cannot enter positive-only Dice")
            truth = np.zeros(shape, dtype=bool)
            for value in encoded:
                truth |= decode_relative_rle(value.strip(), shape)
        source_mask = row.get("mask", row.get("mask_path"))
        if source_mask:
            digest = sha256(source_mask)
            if row.get("mask_sha256") and row["mask_sha256"] != digest:
                raise ValueError(f"Frozen source mask changed: {source_mask}")
            with Image.open(source_mask) as im:
                source = np.asarray(im.convert("L")) > 0
            if source.shape != shape:
                raise ValueError("Source mask and image dimensions differ; no silent GT resizing")
            if truth is not None and not np.array_equal(truth, source):
                raise ValueError(f"Author RLE and source mask differ: {image_path}")
            truth = source if truth is None else truth
            audit["source_files"].append({"path": source_mask, "sha256": digest})
        if truth is None or not truth.any():
            raise ValueError("Positive-only foreground Dice requires nonempty masks")
        masks.append(truth)
    if len({a.shape for a in masks}) != 1:
        raise ValueError("QaTa positive mask dimensions must be identical")
    stacked = np.stack(masks)
    audit.update(shape=list(stacked.shape), foreground_pixels=int(stacked.sum()),
                 boolean_array_sha256=hashlib.sha256(stacked.tobytes()).hexdigest())
    return stacked, audit


def load_maps(path: str | Path, rows: list[dict]) -> tuple[dict[str, np.ndarray], dict]:
    p = Path(path).resolve()
    if p.is_dir():
        p = p / "native_maps.npz"
    expected_paths = [row_image(r) for r in rows]
    expected_patients = [row_patient(r) for r in rows]
    proof = {"path": str(p), "sha256": sha256(p)}
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            paths = data["image_paths"].astype(str).tolist()
            patients = data["patient_ids"].astype(str).tolist()
            class_names = data["class_names"].astype(str).tolist()
            maps = {branch: np.array(data[branch], dtype=np.float32) for branch in BRANCHES}
    else:
        manifest = json.loads(p.read_text())
        paths = manifest["image_paths"]
        patients = manifest.get("patient_ids", manifest.get("group_ids"))
        class_names = manifest.get("class_names", ["covid"])
        descriptors = manifest.get("branches", manifest.get("maps"))
        maps, sources = {}, {}
        for branch in BRANCHES:
            descriptor = descriptors[branch]
            source = Path(descriptor["path"])
            if not source.is_absolute():
                source = p.parent / source
            digest = sha256(source)
            if digest != descriptor["sha256"]:
                raise ValueError(f"Frozen native-map SHA256 mismatch: {branch}")
            maps[branch] = np.load(source, allow_pickle=False, mmap_mode="r")
            sources[branch] = {"path": str(source), "sha256": digest}
        proof["sources"] = sources
    if paths != expected_paths or patients != expected_patients or class_names != ["covid"]:
        raise ValueError("Map identity/order/class/patient IDs differ from frozen split")
    shape = (len(rows), 1, 1370)
    for branch, attention in maps.items():
        if attention.shape != shape or attention.dtype != np.float32 or not np.isfinite(attention).all():
            raise ValueError(f"Unexpected native FP32 map shape/dtype/nonfinite value: {branch}")
    proof["image_count"] = len(rows)
    proof["shape"] = list(shape)
    return maps, proof


def probabilities(attention: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = torch.as_tensor(np.array(attention, dtype=np.float32, copy=True))
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError("Expected [N,1,T] native maps")
    minimum = array.amin(dim=-1, keepdim=True)
    array = (array - minimum) / (array.amax(dim=-1, keepdim=True) - minimum + 1e-8)
    if array.shape[-1] in (1370, 257, 325, 197):
        array = array[..., 1:]
    side = math.isqrt(array.shape[-1])
    if side * side != array.shape[-1]:
        raise ValueError("Non-square native spatial grid")
    heat = array.reshape(-1, 1, side, side)
    return torch.sigmoid(F.interpolate(heat, size=shape, mode="bilinear", align_corners=False))[:, 0].numpy()


def dice_from_probabilities(prob: np.ndarray, truth: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Histograms exactly implement float32 strict >, including threshold ties."""
    thresholds32 = np.asarray(thresholds, dtype=np.float32)
    if len(thresholds32) == 0 or np.any(np.diff(thresholds32) <= 0):
        raise ValueError("Thresholds must be nonempty and strictly increasing")
    p = np.asarray(prob, dtype=np.float32).reshape(len(prob), -1)
    y = np.asarray(truth, dtype=bool).reshape(len(prob), -1)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any(y.sum(1) == 0):
        raise ValueError("Finite matching predictions and nonempty positive masks required")
    result = np.empty((len(p), len(thresholds32)), dtype=np.float32)
    for i, (pred, target) in enumerate(zip(p, y)):
        bins = np.searchsorted(thresholds32, pred, side="left")
        predicted_count = np.cumsum(np.bincount(bins, minlength=len(thresholds32) + 1)[::-1])[::-1][1:]
        intersection = np.cumsum(np.bincount(bins[target], minlength=len(thresholds32) + 1)[::-1])[::-1][1:]
        result[i] = (2 * intersection).astype(np.float32) / (predicted_count + int(target.sum())).astype(np.float32)
    return result


def score_maps(attention: np.ndarray, masks: np.ndarray, thresholds: np.ndarray, batch_size: int = 32) -> np.ndarray:
    if len(attention) != len(masks):
        raise ValueError("Map/mask count mismatch")
    result = np.empty((len(masks), len(thresholds)), dtype=np.float32)
    for start in range(0, len(masks), batch_size):
        stop = min(start + batch_size, len(masks))
        prob = probabilities(attention[start:stop], tuple(masks.shape[-2:]))
        result[start:stop] = dice_from_probabilities(prob, masks[start:stop], thresholds)
    return result


def author_mean(scores: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(scores.T)).nanmean(dim=1).numpy()


def patient_bootstrap(scores: np.ndarray, patient_ids: list[str], replicates: int = 2000,
                      seed: int = 20261024) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resample patients, retaining all their images, then compute image-mean Dice."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(patient_ids) or not np.isfinite(x).all():
        raise ValueError("Expected finite [images, branches] scores and patient IDs")
    patients, inverse = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=x[:, j]) for j in range(x.shape[1])], axis=1)
    draws = np.random.default_rng(seed).integers(0, len(patients), size=(replicates, len(patients)), dtype=np.int32)
    values = np.empty((replicates, x.shape[1]), dtype=np.float64)
    for start in range(0, replicates, 64):
        selection = draws[start:start + 64]
        values[start:start + 64] = sums[selection].sum(1) / counts[selection].sum(1)[:, None]
    return values, draws, patients.tolist()
