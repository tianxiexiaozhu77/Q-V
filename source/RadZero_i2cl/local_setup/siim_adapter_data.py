#!/usr/bin/env python3
"""Prepare leakage-free SIIM adapter splits without changing released data.

The released CSVs split annotation rows, so some images occur in several
splits. Keep the original test image order and labels, then give validation
priority over training. Verify image, patient and study disjointness from DICOM
headers before creating any manifests. This is supervised downstream adaptation,
not a new zero-shot result.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pydicom

from infer import ASSETS_DIR


SPLITS = ("train", "valid", "test")
DEFAULT_OUTPUT = ASSETS_DIR / "splits" / "siim_adapter_v1"
FIELDS = ("index", "image_id", "image_path", "label", "patient_id", "study_id")
EXPECTED_COUNTS = {"train": 7303, "valid": 1668, "test": 1704}
RULES = {
    "version": 1,
    "image_grouping": "first ImageId occurrence order; max binary mask label",
    "label_validation": "row Label must agree with EncodedPixels; overlapping image labels must agree",
    "test": "keep every unique image in released test.csv",
    "valid": "keep released valid.csv images excluding all test images",
    "train": "keep released train.csv images excluding all original valid.csv and test.csv images",
    "metadata_validation": "DICOM SOPInstanceUID equals ImageId; PatientID and StudyInstanceUID are required",
    "leakage_validation": "no ImageId, image_path, PatientID or StudyInstanceUID intersects cleaned splits",
    "preprocessing": "released SIIM read_from_dicom conversion, then released image processor",
    "training_scope": "supervised SIIM adaptation; original images, labels and test split unchanged",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow({"index": index, **row})
    return buffer.getvalue().encode("utf-8")


def _intersections(groups):
    return {
        f"{left}_{right}": {
            key: len({row[key] for row in groups[left]} & {row[key] for row in groups[right]})
            for key in ("image_id", "image_path", "patient_id", "study_id")
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
    }


def _build_expected():
    root = (ASSETS_DIR / "data").resolve()
    sources, grouped, unique, source_counts = {}, {}, {}, {}
    for split in SPLITS:
        path = root / "SIIM" / f"{split}.csv"
        sources[split] = {"path": str(path), "sha256": _sha256(path)}
        by_image = {}
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            image_id = row["ImageId"].strip()
            image_path = row["dicom_path"].strip()
            relative = Path(image_path)
            if not image_id or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe or empty SIIM image reference in {path}: {image_path!r}")
            if relative.stem != image_id or relative.suffix.lower() != ".dcm":
                raise ValueError(f"SIIM image filename does not match ImageId: {image_id}")
            raw_label = float(row["Label"])
            mask = row[" EncodedPixels"].strip()
            label = int(mask != "-1")
            if not mask or raw_label not in (0.0, 1.0) or raw_label != label:
                raise ValueError(f"Invalid or conflicting mask/class label in {path}: {image_id}")
            if image_id in by_image:
                if by_image[image_id]["image_path"] != image_path:
                    raise ValueError(f"Several image paths for {image_id} in {path}")
                by_image[image_id]["label"] = max(by_image[image_id]["label"], label)
            else:
                by_image[image_id] = {"image_id": image_id, "image_path": image_path, "label": label}
        for image_id, row in by_image.items():
            if image_id in unique and unique[image_id] != row:
                raise ValueError(f"Image path or aggregated label conflicts across source splits: {image_id}")
            unique[image_id] = row.copy()
        grouped[split] = by_image
        source_counts[split] = {
            "annotation_rows": len(rows),
            "unique_images": len(by_image),
            "positive_images": sum(row["label"] for row in by_image.values()),
        }

    # Read actual patient/study metadata, never infer patient identity from a path.
    for image_id, row in unique.items():
        image = root / row["image_path"]
        if not image.is_file():
            raise FileNotFoundError(f"Missing SIIM image: {image}")
        dicom = pydicom.dcmread(
            image, stop_before_pixels=True,
            specific_tags=["SOPInstanceUID", "PatientID", "StudyInstanceUID"],
        )
        sop_id = str(getattr(dicom, "SOPInstanceUID", "")).strip()
        patient_id = str(getattr(dicom, "PatientID", "")).strip()
        study_id = str(getattr(dicom, "StudyInstanceUID", "")).strip()
        if sop_id != image_id:
            raise ValueError(f"DICOM SOPInstanceUID does not match ImageId: {image}")
        if not patient_id or not study_id:
            raise ValueError(f"DICOM patient/study metadata missing: {image}")
        row.update(patient_id=patient_id, study_id=study_id)

    original_sets = {split: set(grouped[split]) for split in SPLITS}
    blocked = {
        "test": set(),
        "valid": original_sets["test"],
        "train": original_sets["valid"] | original_sets["test"],
    }
    cleaned = {
        split: [unique[image_id].copy() for image_id in grouped[split] if image_id not in blocked[split]]
        for split in SPLITS
    }
    intersections = _intersections(cleaned)
    if any(count for pair in intersections.values() for count in pair.values()):
        raise ValueError(f"Cleaned SIIM splits contain image/patient/study leakage: {intersections}")
    counts = {
        split: {
            "images": len(rows),
            "positive_images": sum(row["label"] for row in rows),
            "negative_images": sum(1 - row["label"] for row in rows),
            "patients": len({row["patient_id"] for row in rows}),
            "studies": len({row["study_id"] for row in rows}),
            "removed_images": len(grouped[split]) - len(rows),
        }
        for split, rows in cleaned.items()
    }
    actual_counts = {split: count["images"] for split, count in counts.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Released source split counts changed: {actual_counts} != {EXPECTED_COUNTS}")
    if set().union(*({r["image_id"] for r in rows} for rows in cleaned.values())) != set(unique):
        raise ValueError("Cleaned split union does not match original unique images")

    artifacts = {f"{split}.csv": _csv_bytes(cleaned[split]) for split in SPLITS}
    csv_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}
    payload = {
        "schema_version": 1,
        "dataset": "SIIM",
        "data_root": str(root),
        "class_names": ["Pneumothorax"],
        "prompts": ["There is Pneumothorax"],
        "rules": RULES,
        "source_files": sources,
        "source_counts": source_counts,
        "counts": counts,
        "csv_sha256": csv_hashes,
        "original_image_intersections": {
            f"{left}_{right}": len(original_sets[left] & original_sets[right])
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
        },
        "cleaned_intersections": intersections,
        "unique_images_total": len(unique),
        "dicom_headers_validated": len(unique),
        "test_order_and_labels_preserved": cleaned["test"] == [unique[i] for i in grouped["test"]],
        "hash_scope": "SHA256 of original CSVs and generated split CSVs; DICOM headers verified, pixel payloads not hashed",
    }
    manifest = {**payload, "fingerprint": _fingerprint(payload)}
    validation = {
        "status": "passed",
        "errors": [],
        "fingerprint": manifest["fingerprint"],
        "source_csv_sha256": {split: item["sha256"] for split, item in sources.items()},
        "cleaned_csv_sha256": csv_hashes,
        "counts": counts,
        "intersections": intersections,
        "image_files_exist": True,
        "sop_instance_uid_matches": True,
        "nonempty_patient_and_study_ids": True,
        "test_order_and_labels_preserved": True,
        "unique_dicom_headers_checked": len(unique),
    }
    artifacts["split_manifest.json"] = _json_bytes(manifest)
    artifacts["validation.json"] = _json_bytes(validation)
    return manifest, artifacts


def _check_artifacts(output_dir, artifacts, require_all):
    if not output_dir.is_dir():
        if require_all:
            raise FileNotFoundError(f"SIIM split directory does not exist: {output_dir}")
        return
    unexpected = sorted(p.name for p in output_dir.iterdir() if p.name not in {*artifacts, ".prepare.lock"})
    if unexpected:
        raise FileExistsError(f"Unexpected files in split directory; use a fresh directory: {unexpected}")
    for filename, content in artifacts.items():
        path = output_dir / filename
        if path.is_symlink():
            raise FileExistsError(f"Refusing a symlink in split output: {path}")
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(f"Existing split artifact conflicts with current sources: {path}")
        elif require_all:
            raise FileNotFoundError(f"Missing split artifact: {path}")


def prepare_splits(output_dir=DEFAULT_OUTPUT):
    """Return a deterministic manifest; reuse identical files, reject conflicts."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest, artifacts = _build_expected()
    # Reject an unrelated existing directory before even creating the lock file.
    _check_artifacts(output_dir, artifacts, require_all=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_artifacts(output_dir, artifacts, require_all=False)
        for filename, content in artifacts.items():
            destination = output_dir / filename
            if destination.exists():
                continue
            descriptor, temporary_name = tempfile.mkstemp(prefix=".prepare-", dir=output_dir)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        _check_artifacts(output_dir, artifacts, require_all=True)
    return manifest


def validate_splits(output_dir=DEFAULT_OUTPUT):
    """Re-read source CSVs and DICOM headers and compare every generated artifact."""
    output_dir = Path(output_dir).expanduser().resolve()
    manifest, artifacts = _build_expected()
    _check_artifacts(output_dir, artifacts, require_all=True)
    return manifest


def load_split(split_dir, name):
    """Read one generated split in a format compatible with the batch image loader."""
    if name not in SPLITS:
        raise ValueError(f"Unknown SIIM split {name!r}; expected one of {SPLITS}")
    split_dir = Path(split_dir).expanduser().resolve()
    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if _fingerprint(manifest) != fingerprint:
        raise ValueError("SIIM split manifest fingerprint mismatch")
    manifest["fingerprint"] = fingerprint
    validation = json.loads((split_dir / "validation.json").read_text())
    if validation.get("status") != "passed" or validation.get("fingerprint") != fingerprint:
        raise ValueError("SIIM split validation is missing or inconsistent")
    for source in manifest["source_files"].values():
        if _sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Original SIIM source CSV changed: {source['path']}")
    for filename, expected_hash in manifest["csv_sha256"].items():
        if _sha256(split_dir / filename) != expected_hash:
            raise ValueError(f"Generated SIIM split CSV changed: {filename}")
    with (split_dir / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected SIIM split CSV fields")
        rows = list(reader)
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("SIIM split indices are not consecutive")
    if len(rows) != manifest["counts"][name]["images"]:
        raise ValueError("SIIM split row count does not match manifest")
    image_paths = [row["image_path"] for row in rows]
    labels = np.asarray([[int(row["label"])] for row in rows], dtype=np.int8)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("SIIM split labels are not binary")
    data_root = Path(manifest["data_root"])
    for path in image_paths:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts or not (data_root / path).is_file():
            raise ValueError(f"Invalid or missing SIIM image path: {path}")
    return {
        "name": "SIIM",
        "split": name,
        "image_paths": image_paths,
        "labels": labels,
        "class_names": manifest["class_names"],
        "prompts": manifest["prompts"],
        "ids": [row["image_id"] for row in rows],
        "patient_ids": [row["patient_id"] for row in rows],
        "study_ids": [row["study_id"] for row in rows],
        "source_files": [item["path"] for item in manifest["source_files"].values()]
        + [str(split_dir / f"{name}.csv"), str(split_dir / "split_manifest.json")],
        "metadata": {
            "data_root": str(data_root),
            "split_dir": str(split_dir),
            "fingerprint": fingerprint,
            "counts": manifest["counts"][name],
            "rules": manifest["rules"],
            "evaluation_unit": "image",
            "label_encoding": "0=absent, 1=present",
            "split": name,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = validate_splits(args.output) if args.validate_only else prepare_splits(args.output)
    for name in SPLITS:
        loaded = load_split(args.output, name)
        if int(loaded["labels"].sum()) != manifest["counts"][name]["positive_images"]:
            raise ValueError(f"Loaded labels differ from manifest: {name}")
    print(json.dumps({
        "status": "passed",
        "output": str(args.output.expanduser().resolve()),
        "fingerprint": manifest["fingerprint"],
        "counts": manifest["counts"],
        "cleaned_intersections": manifest["cleaned_intersections"],
    }, indent=2))


if __name__ == "__main__":
    main()
