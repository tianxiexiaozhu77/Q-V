#!/usr/bin/env python3
"""Fixed grouped Node21 development splits for supervised adapter experiments.

Only 1,134 positive images have a source-name mapping; patient identities are
unavailable for most images. The resulting custom split is image/group disjoint,
but must not be described as a fully verified patient-disjoint clinical test set.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import cv2
import numpy as np

from infer import ASSETS_DIR
from node21_data import DEFAULT_ROOT, PREPROCESSING_PROTOCOL, load_node21, read_node21_pixels


DEFAULT_OUTPUT = ASSETS_DIR / "splits/node21_adapter_v1"
SPLITS = ("train", "valid", "test")
SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)
FIELDS = (
    "index", "image_id", "image_path", "label", "origin_dataset",
    "original_image_name", "patient_id", "study_id", "file_sha256",
    "pixel_sha256", "normalized_uint8_sha256", "phash64", "group_id",
)
RULES = {
    "version": 1,
    "seed": SEED,
    "target_fractions": dict(zip(SPLITS, FRACTIONS)),
    "label_source": "real processed metadata.csv, one binary label per image; simulated_metadata.csv is excluded",
    "scope": "custom held-out split of the public Node21 development collection, not the hidden challenge test",
    "known_patient_grouping": "chestxray14 source filename: eight-digit patient prefix before underscore",
    "known_study_grouping": "openi source filename: report/study prefix before _IM-; no claim this is a patient ID",
    "unknown_patient_grouping": "empty patient_id is unknown, never a fabricated unique patient identity",
    "duplicate_grouping": "transitive union of known patient/study identifiers, identical decoded pixels, identical normalized uint8 pixels, and conservative near-duplicate matches",
    "near_duplicate_screen": "63 low-frequency pHash bits (DC excluded), Hamming distance <=4, then normalized grayscale 64x64 and 128x128 Pearson correlation >=0.9995 and RMSE <=0.01",
    "near_duplicate_limit": "conservative similarity screen; not a guarantee that every crop, mirror, or repeat acquisition was recognized",
    "assignment": "seeded greedy group assignment minimizing normalized squared source-by-label count deviations from 70/15/15 targets; seed-shuffled order within each split",
    "preprocessing": PREPROCESSING_PROTOCOL,
    "training_scope": "frozen RadZero plus a supervised downstream adapter; no official test labels are available",
    "patient_disjointness_guaranteed": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow({"index": index, **{key: row[key] for key in FIELDS if key != "index"}})
    return stream.getvalue().encode()


def _mha_metadata_keys(path):
    keys = []
    with Path(path).open("rb") as stream:
        for _ in range(128):
            line = stream.readline(8192).decode("ascii").strip()
            if "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            keys.append(key)
            if key == "ElementDataFile":
                return keys
    raise ValueError(f"Incomplete MHA header: {path}")


def _inspect_image(path):
    cv2.setNumThreads(0)
    pixels = read_node21_pixels(path)
    byte_ordered = np.ascontiguousarray(pixels.astype(pixels.dtype.newbyteorder("<"), copy=False))
    pixel_digest = hashlib.sha256()
    pixel_digest.update(str(byte_ordered.dtype).encode())
    pixel_digest.update(str(byte_ordered.shape).encode())
    pixel_digest.update(byte_ordered.tobytes())
    normalized = cv2.normalize(pixels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    small32 = cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small32)[:8, :8].reshape(-1)[1:]
    bits = coefficients > np.median(coefficients)
    phash = sum(int(bit) << position for position, bit in enumerate(bits))
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": pixel_digest.hexdigest(),
        "normalized_uint8_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
        "phash64": f"{phash:016x}",
        "_thumbnail": cv2.resize(normalized, (64, 64), interpolation=cv2.INTER_AREA),
        "_metadata_keys": _mha_metadata_keys(path),
    }


class _UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _similarity(left, right):
    left, right = left.astype(np.float64).reshape(-1) / 255, right.astype(np.float64).reshape(-1) / 255
    rmse = float(np.sqrt(np.mean((left - right) ** 2)))
    centered_left, centered_right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
    correlation = float(np.dot(centered_left, centered_right) / denominator) if denominator else 0.0
    return correlation, rmse


def _thumbnail128(path):
    pixels = read_node21_pixels(path)
    normalized = cv2.normalize(pixels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.resize(normalized, (128, 128), interpolation=cv2.INTER_AREA)


def _intersections(groups):
    keys = ("image_id", "image_path", "patient_id", "study_id", "group_id",
            "file_sha256", "pixel_sha256", "normalized_uint8_sha256")
    return {
        f"{left}_{right}": {
            key: len({row[key] for row in groups[left] if row[key]} &
                     {row[key] for row in groups[right] if row[key]})
            for key in keys
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }


def _assign_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    categories = sorted({(row["origin_dataset"], row["label"]) for row in rows})
    category_index = {key: index for index, key in enumerate(categories)}
    totals = np.zeros(len(categories), dtype=np.int64)
    entries = []
    random = np.random.default_rng(SEED)
    for group_id in sorted(grouped):
        group = grouped[group_id]
        vector = np.zeros(len(categories), dtype=np.int64)
        for row in group:
            vector[category_index[(row["origin_dataset"], row["label"])]] += 1
        totals += vector
        entries.append((group_id, group, vector, float(random.random())))
    entries.sort(key=lambda item: (-len(item[1]), item[3], item[0]))
    targets = np.asarray(FRACTIONS)[:, None] * totals[None, :]
    assigned = np.zeros_like(targets)
    result = {name: [] for name in SPLITS}
    for _, group, vector, _ in entries:
        deltas = (((assigned + vector - targets) ** 2 -
                   (assigned - targets) ** 2) / np.maximum(targets, 1)).sum(axis=1)
        position = int(np.argmin(deltas))
        result[SPLITS[position]].extend(group)
        assigned[position] += vector
    for position, split in enumerate(SPLITS):
        permutation = np.random.default_rng(SEED + 100 + position).permutation(len(result[split]))
        result[split] = [result[split][index] for index in permutation]
        for limit in (64, 128):
            if len({row["label"] for row in result[split][:limit]}) != 2:
                raise ValueError(f"Smoke prefix lacks both classes: {split}:{limit}")
    return result


def _build_expected(workers=8):
    spec = load_node21()
    mapping_path = DEFAULT_ROOT / "cxr_images/original_data/filenames_orig_and_new.csv"
    with mapping_path.open(newline="") as stream:
        mappings = list(csv.DictReader(stream))
    by_id = {}
    for item in mappings:
        key = item["node21_img_id"]
        if key in by_id:
            raise ValueError(f"Repeated source mapping: {key}")
        by_id[key] = item
    if set(by_id) != {image_id for image_id, label in zip(spec["ids"], spec["labels"]) if int(label[0])}:
        raise ValueError("Expected the released source mapping to cover exactly positive images")
    rows = []
    for image_id, path, label in zip(spec["ids"], spec["image_paths"], spec["labels"]):
        mapping = by_id.get(image_id, {})
        source = mapping.get("orig_dataset", "unknown")
        original = mapping.get("original_image_name", "")
        patient, study = "", ""
        if source == "chestxray14":
            if not re.fullmatch(r"\d{8}_\d{3}", original):
                raise ValueError(f"Unexpected NIH source filename: {original}")
            patient = "chestxray14:" + original.split("_")[0]
        elif source == "openi":
            match = re.fullmatch(r"(\d+)_IM-.+", original)
            if not match:
                raise ValueError(f"Unexpected OpenI source filename: {original}")
            study = "openi:" + match.group(1)
        rows.append({
            "image_id": image_id, "image_path": path, "label": int(label[0]),
            "origin_dataset": source, "original_image_name": original,
            "patient_id": patient, "study_id": study,
        })
    metadata_keys = Counter()
    thumbnails = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, inspected in zip(rows, pool.map(_inspect_image, spec["image_paths"])):
            thumbnails.append(inspected.pop("_thumbnail"))
            metadata_keys.update(inspected.pop("_metadata_keys"))
            row.update(inspected)
    if any(re.search(r"patient|studyinstance", key, flags=re.IGNORECASE) for key in metadata_keys):
        raise ValueError("MHA headers contain additional patient metadata; review it before splitting")
    union = _UnionFind(len(rows))
    linkage_counts = {}
    exact_groups = {}
    for key in ("patient_id", "study_id", "file_sha256", "pixel_sha256", "normalized_uint8_sha256"):
        lookup = defaultdict(list)
        for index, row in enumerate(rows):
            if row[key]:
                lookup[row[key]].append(index)
        repeated = [indices for indices in lookup.values() if len(indices) > 1]
        for indices in repeated:
            for index in indices[1:]:
                union.union(indices[0], index)
        linkage_counts[key] = {"repeated_values": len(repeated),
                               "images_in_repeated_values": sum(map(len, repeated))}
        if "sha256" in key:
            exact_groups[key] = [[rows[index]["image_id"] for index in indices] for indices in repeated]
    hashes = [int(row["phash64"], 16) for row in rows]
    near_pairs, candidate_count, decoded128 = [], 0, {}
    for right in range(len(rows)):
        for left in range(right):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance > 4:
                continue
            candidate_count += 1
            if rows[left]["normalized_uint8_sha256"] == rows[right]["normalized_uint8_sha256"]:
                continue
            correlation, rmse = _similarity(thumbnails[left], thumbnails[right])
            if correlation < 0.9995 or rmse > 0.01:
                continue
            for index in (left, right):
                if index not in decoded128:
                    decoded128[index] = _thumbnail128(rows[index]["image_path"])
            correlation, rmse = _similarity(decoded128[left], decoded128[right])
            if correlation >= 0.9995 and rmse <= 0.01:
                union.union(left, right)
                near_pairs.append({
                    "left": rows[left]["image_id"], "right": rows[right]["image_id"],
                    "phash_hamming": distance, "pearson_128": correlation, "rmse_128": rmse,
                })
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[union.find(index)].append(row["image_id"])
    group_ids = {key: "node21:" + min(value) for key, value in members.items()}
    for index, row in enumerate(rows):
        row["group_id"] = group_ids[union.find(index)]
    split_rows = _assign_groups(rows)
    intersections = _intersections(split_rows)
    if any(count for pair in intersections.values() for count in pair.values()):
        raise ValueError(f"Unexpected cross-split grouping overlap: {intersections}")
    counts = {
        split: {
            "images": len(items), "positive_images": sum(row["label"] for row in items),
            "negative_images": sum(1 - row["label"] for row in items),
            "groups": len({row["group_id"] for row in items}),
            "images_with_known_patient_id": sum(bool(row["patient_id"]) for row in items),
            "known_patients": len({row["patient_id"] for row in items if row["patient_id"]}),
            "images_with_known_study_id": sum(bool(row["study_id"]) for row in items),
            "source_counts": dict(sorted(Counter(row["origin_dataset"] for row in items).items())),
        }
        for split, items in split_rows.items()
    }
    source_paths = [*map(Path, spec["source_files"]), mapping_path,
                    Path(__file__).resolve(), Path(__file__).with_name("node21_data.py").resolve()]
    source_files = {f"source_{index}": {"path": str(path), "sha256": _sha256(path)}
                    for index, path in enumerate(source_paths)}
    duplicate_audit = {
        "status": "passed", "linkage_counts": linkage_counts,
        "exact_duplicate_groups": exact_groups, "near_duplicate_candidate_pairs": candidate_count,
        "accepted_near_duplicate_pairs": near_pairs,
        "groups": len(members), "maximum_group_size": max(map(len, members.values())),
        "multi_image_groups": [sorted(items) for items in members.values() if len(items) > 1],
        "limitations": RULES["near_duplicate_limit"],
    }
    artifacts = {f"{split}.csv": _csv_bytes(split_rows[split]) for split in SPLITS}
    artifacts["image_fingerprints.csv"] = _csv_bytes(rows)
    artifacts["duplicate_audit.json"] = _json_bytes(duplicate_audit)
    patient_coverage = {
        "total_images": len(rows),
        "source_mapping_images": len(by_id),
        "images_with_known_patient_id": sum(bool(row["patient_id"]) for row in rows),
        "images_without_known_patient_id": sum(not bool(row["patient_id"]) for row in rows),
        "known_patient_ids": len({row["patient_id"] for row in rows if row["patient_id"]}),
        "images_with_known_study_id": sum(bool(row["study_id"]) for row in rows),
        "mha_header_field_counts": dict(sorted(metadata_keys.items())),
        "mha_patient_fields_present": False,
        "patient_disjointness_guaranteed": False,
        "limitation": "The released mapping covers positive images only; negative-image source/patient IDs and PadChest/JSRT patient IDs are unavailable. Only known NIH patients and OpenI studies can be explicitly grouped.",
    }
    payload = {
        "schema_version": 1, "dataset": "Node21", "data_root": str(DEFAULT_ROOT),
        "class_names": ["Nodule"], "prompts": spec["prompts"], "rules": RULES,
        "source_files": source_files, "counts": counts, "unique_images_total": len(rows),
        "csv_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in artifacts.items() if name.endswith(".csv")},
        "artifact_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in artifacts.items()},
        "cleaned_intersections": intersections, "patient_metadata_coverage": patient_coverage,
        "duplicate_summary": {key: value for key, value in duplicate_audit.items()
                              if key not in ("exact_duplicate_groups", "multi_image_groups", "accepted_near_duplicate_pairs")},
        "accepted_near_duplicate_pairs": len(near_pairs),
        "pretrained_data_overlap_not_excluded": "Source data include ChestXray14/OpenI/PadChest; no claim of independence from RadZero pretraining data.",
        "hash_scope": "Every source MHA file, decoded pixel array, normalized uint8 image, source metadata, split helper code, and generated split CSV is SHA256 fingerprinted.",
    }
    manifest = {**payload, "fingerprint": _fingerprint(payload)}
    validation = {
        "status": "passed", "errors": [], "fingerprint": manifest["fingerprint"],
        "counts": counts, "intersections": intersections,
        "all_4882_images_in_exactly_one_split": len(rows) == 4882,
        "all_real_labels_preserved": True, "simulated_metadata_used": False,
        "known_patient_and_study_groups_disjoint": True,
        "patient_disjointness_guaranteed": False,
        "pixel_and_file_fingerprints_computed": len(rows),
    }
    artifacts["split_manifest.json"] = _json_bytes(manifest)
    artifacts["validation.json"] = _json_bytes(validation)
    return manifest, artifacts


def _read_verified(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = json.loads((output_dir / "split_manifest.json").read_text())
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if _fingerprint(payload) != manifest["fingerprint"]:
        raise ValueError("Node21 split manifest fingerprint mismatch")
    if manifest["rules"] != RULES:
        raise ValueError("Node21 split rules differ from this helper version")
    for source in manifest["source_files"].values():
        if _sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Node21 source file changed: {source['path']}")
    for filename, expected in manifest["artifact_sha256"].items():
        if _sha256(output_dir / filename) != expected:
            raise ValueError(f"Generated Node21 artifact changed: {filename}")
    validation = json.loads((output_dir / "validation.json").read_text())
    if validation.get("status") != "passed" or validation.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("Node21 split validation is missing or inconsistent")
    return manifest


def validate_splits(output_dir=DEFAULT_OUTPUT, workers=8):
    """Recheck source/artifact hashes, image bytes, labels, coverage, and group separation."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = _read_verified(output_dir)
    spec = load_node21()
    truth = dict(zip(spec["image_paths"], map(int, spec["labels"][:, 0])))
    all_rows, grouped = [], {}
    for split in SPLITS:
        with (output_dir / f"{split}.csv").open(newline="") as stream:
            grouped[split] = list(csv.DictReader(stream))
        all_rows.extend(grouped[split])
    paths = [row["image_path"] for row in all_rows]
    if len(paths) != len(set(paths)) or set(paths) != set(truth):
        raise ValueError("Node21 split coverage is not exactly the real image collection")
    if any(int(row["label"]) != truth[row["image_path"]] for row in all_rows):
        raise ValueError("Node21 split label differs from original real metadata")
    intersections = _intersections(grouped)
    if intersections != manifest["cleaned_intersections"] or any(
            value for pair in intersections.values() for value in pair.values()):
        raise ValueError("Node21 grouping intersection audit failed")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row, digest in zip(all_rows, pool.map(_sha256, paths)):
            if row["file_sha256"] != digest:
                raise ValueError(f"Node21 MHA bytes changed: {row['image_path']}")
    return manifest


def prepare_splits(output_dir=DEFAULT_OUTPUT, validate_only=False, workers=8):
    """Create immutable deterministic split artifacts or validate an existing set."""
    output_dir = Path(output_dir).expanduser().resolve()
    if validate_only or (output_dir / "split_manifest.json").exists():
        return validate_splits(output_dir, workers)
    if output_dir.exists() and any(path.name != ".prepare.lock" for path in output_dir.iterdir()):
        raise FileExistsError(f"Split directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output_dir / "split_manifest.json").exists():
            return validate_splits(output_dir, workers)
        manifest, artifacts = _build_expected(workers)
        for filename, value in artifacts.items():
            destination = output_dir / filename
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite split artifact: {destination}")
            descriptor, name = tempfile.mkstemp(prefix=".prepare-", dir=output_dir)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
                Path(name).replace(destination)
            finally:
                if Path(name).exists():
                    Path(name).unlink()
    return manifest


def load_split(split_dir, name):
    """Load verified split metadata; returned image paths are absolute."""
    if name not in SPLITS:
        raise ValueError(f"Unknown Node21 split: {name}")
    split_dir = Path(split_dir).expanduser().resolve()
    manifest = _read_verified(split_dir)
    with (split_dir / f"{name}.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected Node21 split columns")
        rows = list(reader)
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Node21 split indices are not consecutive")
    labels = np.asarray([[int(row["label"])] for row in rows], dtype=np.int8)
    if len(rows) != manifest["counts"][name]["images"] or not np.isin(labels, (0, 1)).all():
        raise ValueError("Node21 split row count or labels are invalid")
    paths = [row["image_path"] for row in rows]
    if any(not Path(path).is_absolute() or not Path(path).is_file() for path in paths):
        raise ValueError("Node21 split contains a missing/nonabsolute image path")
    return {
        "name": "Node21", "split": name, "image_paths": paths, "labels": labels,
        "class_names": manifest["class_names"], "prompts": manifest["prompts"],
        "ids": [row["image_id"] for row in rows],
        "patient_ids": [row["patient_id"] for row in rows],
        "study_ids": [row["study_id"] for row in rows],
        "group_ids": [row["group_id"] for row in rows],
        "file_sha256": [row["file_sha256"] for row in rows],
        "pixel_sha256": [row["pixel_sha256"] for row in rows],
        "normalized_uint8_sha256": [row["normalized_uint8_sha256"] for row in rows],
        "source_files": [item["path"] for item in manifest["source_files"].values()]
        + [str(split_dir / f"{name}.csv"), str(split_dir / "split_manifest.json")],
        "metadata": {
            "data_root": manifest["data_root"], "split_dir": str(split_dir),
            "fingerprint": manifest["fingerprint"], "counts": manifest["counts"][name],
            "rules": manifest["rules"], "patient_metadata_coverage": manifest["patient_metadata_coverage"],
            "evaluation_unit": "image", "label_encoding": "0=absent, 1=present", "split": name,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = prepare_splits(args.output, args.validate_only, args.workers)
    for split in SPLITS:
        loaded = load_split(args.output, split)
        if int(loaded["labels"].sum()) != manifest["counts"][split]["positive_images"]:
            raise ValueError(f"Loaded labels do not match split manifest: {split}")
    print(json.dumps({
        "status": "passed", "output": str(args.output.resolve()),
        "fingerprint": manifest["fingerprint"], "counts": manifest["counts"],
        "patient_metadata_coverage": manifest["patient_metadata_coverage"],
        "duplicate_summary": manifest["duplicate_summary"],
        "accepted_near_duplicate_pairs": manifest["accepted_near_duplicate_pairs"],
        "cleaned_intersections": manifest["cleaned_intersections"],
    }, indent=2))


if __name__ == "__main__":
    main()
