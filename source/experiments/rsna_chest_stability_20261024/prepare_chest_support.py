#!/usr/bin/env python3
"""Freeze five ChestX-Det10 training-support draws and fetch their ZIP members.

The published training annotation is the entire metadata candidate pool. Only
the union of the five selected supports is downloaded, before any prediction.
The existing 492-image classification and 409-image localization tests remain
untouched. No unverified patient grouping is claimed for this dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
SOURCE = HERE / "source_data"
IMAGE_ROOT = MEDICAL / "data/alpharad/ChestXDet10/stability_train_support_20261024"
TRAIN_URL = "https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det10-Dataset/master/train.json"
ZIP_URL = "https://resource.deepwise.com/xraychallenge/train_data.zip"
ZIP_BYTES = 1210114616
SEEDS = [42, 43, 44, 45, 46]
CLASSES = ["Atelectasis", "Calcification", "Consolidation", "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"Refusing to alter frozen source: {path}")
    if not path.exists():
        path.write_bytes(data)


def write_json(path, obj):
    frozen(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def fetch(url, start=None, end=None):
    for attempt in range(5):
        try:
            headers = {"User-Agent": "Medical-native-query-stability-research/1.0"}
            if start is not None:
                headers["Range"] = f"bytes={start}-{end - 1}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                data = response.read()
                if start is not None:
                    expected = f"bytes {start}-{end - 1}/{ZIP_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected or len(data) != end - start:
                        raise ValueError(f"Range response differs: {response.status} {response.headers.get('Content-Range')} expected {expected}")
                return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


def remote_index():
    tail = fetch(ZIP_URL, ZIP_BYTES - 65557, ZIP_BYTES)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise ValueError("ZIP EOCD not found")
    fields = list(struct.unpack("<4s4H2LH", tail[offset:offset + 22]))
    central_bytes, central_offset = fields[5:7]
    central = fetch(ZIP_URL, central_offset, central_offset + central_bytes)
    fields[6] = 0  # Only central directory is in this in-memory ZIP.
    fake = central + struct.pack("<4s4H2LH", *fields) + tail[offset + 22:offset + 22 + fields[7]]
    infos = zipfile.ZipFile(io.BytesIO(fake)).infolist()
    index = {}
    for info in infos:
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if name in index:
            raise ValueError("Duplicate filename in archive")
        index[name] = {"archive_name": info.filename, "offset": info.header_offset,
                       "compressed_bytes": info.compress_size, "bytes": info.file_size,
                       "crc32": info.CRC, "compression": info.compress_type}
    return index


def download_member(item):
    name, info = item
    dest = IMAGE_ROOT / name
    if dest.exists():
        data = dest.read_bytes()
    else:
        header = fetch(ZIP_URL, info["offset"], info["offset"] + 30)
        fields = struct.unpack("<4s5H3L2H", header)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError("Invalid/encrypted ZIP local header")
        name_size, extra_size = fields[-2:]
        start = info["offset"] + 30
        payload = fetch(ZIP_URL, start, start + name_size + extra_size + info["compressed_bytes"])
        if payload[:name_size].decode() != info["archive_name"]:
            raise ValueError("Archive entry name mismatch")
        packed = payload[name_size + extra_size:]
        if info["compression"] == 8:
            data = zlib.decompress(packed, -15)
        elif info["compression"] == 0:
            data = packed
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(data) != info["bytes"] or zlib.crc32(data) != info["crc32"]:
            raise ValueError("Downloaded image size/CRC mismatch")
        frozen(dest, data)
    if len(data) != info["bytes"] or zlib.crc32(data) != info["crc32"]:
        raise ValueError("Cached image size/CRC mismatch")
    return name, image_identity(dest)


def image_identity(path):
    path = Path(path)
    with Image.open(path) as im:
        image = im.convert("RGB")
        payload = str(image.size).encode() + b"\0RGB\0" + image.tobytes()
        return {"image": str(path), "byte_sha256": sha(path), "pixel_sha256": hashlib.sha256(payload).hexdigest(),
                "width": image.width, "height": image.height}


def sample(rows, seed):
    labels = np.asarray([r["labels"] for r in rows], dtype=np.int8)
    candidates = {cls: np.flatnonzero(labels[:, c] == 1).tolist() for c, cls in enumerate(CLASSES)}
    order = sorted(candidates, key=lambda key: (len(candidates[key]), key))
    for attempt in range(512):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        chosen, used = [], set()
        for anchor in order:
            available = [int(i) for i in rng.permutation(candidates[anchor]) if int(i) not in used]
            if len(available) < 5:
                break
            for index in available[:5]:
                chosen.append({**rows[index], "anchor_class": anchor})
                used.add(index)
        if len(chosen) == 50:
            selected_y = np.asarray([r["labels"] for r in chosen])
            if np.all(selected_y.sum(0) > 0) and np.all(selected_y.sum(0) < 50):
                return chosen, {"seed": seed, "deterministic_attempt": attempt, "anchor_order": order,
                                "anchor_counts": dict(Counter(r["anchor_class"] for r in chosen)),
                                "positive_counts": selected_y.sum(0).tolist(), "negative_counts": (50 - selected_y.sum(0)).tolist()}
    raise ValueError("No valid label-only support sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    annotations_path = SOURCE / "chestxdet10_author_train.json"
    if not annotations_path.exists():
        frozen(annotations_path, fetch(TRAIN_URL))
    annotations = json.loads(annotations_path.read_text())
    native = MEDICAL / "experiments/fewshot_native_query/protocol_v1/chestxdet10.json"
    old = json.loads(native.read_text())
    test = json.loads(Path(old["test_manifest"]["path"]).read_text())["rows"]
    old_support = old["support_rows"]
    old_names = {Path(r["image"]).name for r in test + old_support}
    rows, excluded_names = [], []
    for entry in sorted(annotations, key=lambda r: r["file_name"]):
        if entry["file_name"] in old_names:
            excluded_names.append(entry["file_name"])
            continue
        if not set(entry["syms"]) <= set(CLASSES):
            raise ValueError("Unexpected annotation class")
        rows.append({"image_id": Path(entry["file_name"]).stem, "filename": entry["file_name"],
                     "labels": [int(c in entry["syms"]) for c in CLASSES], "source_split": "author_train"})
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicated training filename")
    draws = {str(seed): sample(rows, seed) for seed in SEEDS}
    names = sorted({r["filename"] for selected, _ in draws.values() for r in selected})
    # Freeze the choices before downloading or running any model.
    write_json(SOURCE / "chestxdet10_sampling_freeze.json", {
        "version": "chestxdet10-author-train-five-seeds-v1", "seed": SEEDS,
        "class_names": CLASSES, "candidate_count": len(rows), "candidate_rows": rows,
        "test_manifest_sha256": old["test_manifest"]["sha256"],
        "sampling": "Distinct assignments to five positive anchors per class, rarest class first (count then name), numpy default_rng(SeedSequence([seed,attempt])), full original multilabel vectors; all 50 support labels enter every class's query fit. No background quota.",
        "seeds": {seed: {"support_rows": selected, "sampling": metadata} for seed, (selected, metadata) in draws.items()},
        "annotation_url": TRAIN_URL, "annotation_sha256": sha(annotations_path),
        "excluded_overlap_with_all_original_542_filenames": excluded_names,
        "selected_union_count": len(names), "selection_uses_predictions": False,
        "seed42_is_historical_support": False,
    })
    print(f"FROZEN ChestXDet10 metadata candidates={len(rows)} support_union={len(names)}", flush=True)
    index_path = SOURCE / "chestxdet10_train_zip_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["members"]
    else:
        index = remote_index()
        write_json(index_path, {"url": ZIP_URL, "archive_bytes": ZIP_BYTES, "members": index})
    if set(names) - set(index):
        raise ValueError("Selected image missing in archive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        images = dict(executor.map(download_member, [(name, index[name]) for name in names]))
        test_inventory = list(executor.map(lambda r: image_identity(r["image"]), test))
    test_pixels = {r["pixel_sha256"] for r in test_inventory}
    if len({r["pixel_sha256"] for r in images.values()}) != len(images):
        raise ValueError("Duplicate decoded images in selected training union; protocol must be revised before inference")
    if test_pixels & {r["pixel_sha256"] for r in images.values()}:
        raise ValueError("Selected training/test decoded-image overlap; protocol must be revised before inference")
    final_seeds = {}
    for seed, (selected, metadata) in draws.items():
        support = []
        for row in selected:
            item = {**row, **images[row["filename"]]}
            item["group_id"] = "ChestXDet10:pixel:" + item["pixel_sha256"]
            support.append(item)
        final_seeds[seed] = {"support_rows": support, "sampling": metadata}
    write_json(SOURCE / "chestxdet10_pool.json", {
        "dataset": "chestxdet10", "class_names": CLASSES, "candidate_rows": rows,
        "seeds": final_seeds,
        "sources": {str(p): sha(p) for p in [annotations_path, index_path, SOURCE / "chestxdet10_sampling_freeze.json", Path(__file__).resolve()]},
        "source_urls": ["https://github.com/Deepwise-AILab/ChestX-Det10-Dataset", TRAIN_URL, ZIP_URL],
        "test_image_inventory": test_inventory,
        "audit": {"candidate_metadata_count": len(rows), "downloaded_selected_union_count": len(names),
                  "all_candidate_images_downloaded": False, "downloaded_image_bytes": sum(index[n]["bytes"] for n in names),
                  "zip_member_size_crc32_verified": True, "decoded_duplicate_support_union": 0,
                  "decoded_duplicate_with_test": 0, "patient_identity_available": False,
                  "original_542_filename_overlap": excluded_names,
                  "test_unique_pixel_groups": len(test_pixels),
                  "limitations": ["No verified patient IDs; image/hash separation is not patient separation.",
                                  "Only selected training support images are downloaded and checked, not all metadata candidates.",
                                  "No near-duplicate or pretraining overlap search is implied.",
                                  "New training supports differ from historical test-held-out supports."]},
    })
    print(f"READY {SOURCE / 'chestxdet10_pool.json'}", flush=True)


if __name__ == "__main__":
    main()
