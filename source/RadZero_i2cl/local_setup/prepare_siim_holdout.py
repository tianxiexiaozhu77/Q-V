#!/usr/bin/env python3
"""Prepare the explicitly requested SIIM holdout from available test DICOMs.

Select five positive and five negative support images using this project's
unchanged native selection algorithm. Exclude every support-associated patient,
study and byte-identical file group from evaluation. This is a supervised
few-shot holdout, not the original independent-train/full-test SIIM protocol.
No model, optimization setting, image, annotation or other dataset is changed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from pathlib import Path
import tempfile

import pydicom

from batch_data import load_dataset
from i2cl_data import (
    PROTOCOL_VERSION, _UnionFind, _check_artifacts, _counts, _csv_bytes,
    _fingerprint, _json_bytes, _select_support, _source_records, load_spec, sha256,
)
from infer import ASSETS_DIR


DEFAULT_OUTPUT = ASSETS_DIR / "splits/siim_released_holdout_shot5_seed42"


def prepare(output_dir, shots_per_class=5, seed=42):
    if shots_per_class < 1 or seed < 0:
        raise ValueError("shots_per_class must be positive and seed nonnegative")
    root = (ASSETS_DIR / "data").resolve()
    spec = load_dataset("SIIM", root)
    rows = []
    for path, label in zip(spec["image_paths"], spec["labels"]):
        image_id = Path(path).stem
        header = pydicom.dcmread(
            root / path, stop_before_pixels=True,
            specific_tags=["PatientID", "StudyInstanceUID", "SOPInstanceUID"],
        )
        patient = str(getattr(header, "PatientID", "")).strip()
        study = str(getattr(header, "StudyInstanceUID", "")).strip()
        if not patient or not study:
            raise ValueError(f"Missing SIIM patient/study identity: {path}")
        if str(getattr(header, "SOPInstanceUID", "")).strip() != image_id:
            raise ValueError(f"SIIM SOPInstanceUID differs from filename: {path}")
        rows.append({
            "image_path": path, "image_id": image_id,
            "patient_id": patient, "study_id": study,
            "source_split": "released_test",
            "labels": label.tolist(),
        })

    paths = [row["image_path"] for row in rows]
    print(f"SIIM: hashing {len(rows)} available DICOMs", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha256, [root / path for path in paths]))
    union, representatives = _UnionFind(len(rows)), {}
    for index, (row, digest) in enumerate(zip(rows, hashes)):
        row["byte_sha256"] = digest
        for kind in ("patient_id", "study_id", "byte_sha256"):
            key = (kind, row[kind])
            if key in representatives:
                union.union(index, representatives[key])
            else:
                representatives[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(union.find(index), []).append(index)
    for members in groups.values():
        group = "SIIM:" + _fingerprint(sorted(paths[index] for index in members))[:24]
        for index in members:
            rows[index]["group"] = group

    candidates = list(range(len(rows)))
    support, test, excluded, anchors, selection = _select_support(
        rows, candidates, candidates, spec["class_names"],
        shots_per_class, seed, independent_train=False,
    )
    for index, anchor in anchors.items():
        rows[index]["anchor_class"] = anchor
    splits = {
        "support": [rows[index] for index in support],
        "test": [rows[index] for index in test],
        "excluded": [rows[index] for index in excluded],
        "inventory": rows,
    }
    csv_artifacts = {f"{split}.csv": _csv_bytes(items)
                     for split, items in splits.items()}
    csv_hashes = {name: hashlib.sha256(content).hexdigest()
                  for name, content in csv_artifacts.items()}
    counts = {split: _counts(items, spec["class_names"])
              for split, items in splits.items()}
    intersections = {
        key: len({row[key] for row in splits["support"] if row[key]} &
                 {row[key] for row in splits["test"] if row[key]})
        for key in ("image_path", "group", "patient_id", "study_id", "byte_sha256")
    }
    if any(intersections.values()):
        raise AssertionError(f"SIIM support/test leakage: {intersections}")
    support_sha = csv_hashes["support.csv"]
    protocol = "released_test_fewshot_group_holdout"
    payload = {
        "schema_version": PROTOCOL_VERSION, "name": "SIIM",
        "data_root": str(root), "class_names": spec["class_names"],
        "prompts": spec["prompts"], "seed": seed,
        "shots_per_class": shots_per_class, "protocol": protocol,
        "protocol_exception": (
            "User explicitly selected five positive and five negative support "
            "images from locally available SIIM test DICOMs; independent "
            "training images are unavailable. Support-associated patient, "
            "study and file groups are removed from evaluation."
        ),
        "evaluation_unit": "image",
        "context_and_calibration_use_identical_support": True,
        "support_sha256": support_sha,
        "context_support_sha256": support_sha,
        "calibration_support_sha256": support_sha,
        "support_semantics": (
            "Five distinct images per positive and negative state "
            "(or requested shots_per_class); unchanged native selection."
        ),
        "selection": selection, "counts": counts, "csv_sha256": csv_hashes,
        "support_evaluation_intersections": intersections,
        "removed_from_released_evaluation": len(rows) - len(test),
        "discarded_non_support_same_group_images": len(excluded),
        "support_rows_in_released_evaluation": len(support),
        "evaluation_order": (
            "Released SIIM test order with entire support-associated groups excluded."
        ),
        "label_semantics": spec["metadata"]["label_source"],
        "source_metadata": spec["metadata"],
        "grouping": (
            "PatientID and StudyInstanceUID read from every available SIIM "
            "DICOM header, merged with byte-identical file groups."
        ),
        "limitations": [
            "No independent local SIIM training pool; support is held out of the released test pool.",
            "Results use a supervised few-shot holdout, not the original full-test zero-shot protocol.",
            "Byte-identical files are grouped; visually near-duplicate pixels are not searched.",
        ],
        "source_files": _source_records([
            *spec["source_files"], Path(__file__),
            Path(__file__).with_name("i2cl_data.py"),
            Path(__file__).with_name("batch_data.py"),
        ]),
        "image_content_fingerprint": _fingerprint(list(zip(paths, hashes))),
        "dicom_headers_validated": len(rows),
    }
    manifest = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts = {f"SIIM/{name}": content
                 for name, content in csv_artifacts.items()}
    artifacts["SIIM/manifest.json"] = _json_bytes(manifest)
    root_payload = {
        "schema_version": PROTOCOL_VERSION, "datasets": ["SIIM"],
        "data_root": str(root), "seed": seed, "shots_per_class": shots_per_class,
        "context_and_calibration_use_identical_support": True,
        "independent_model_per_dataset": True,
        "validation_or_test_used_for_coefficient_selection": False,
        "split_selection_uses_labels_only_for_class_quota_and_outcome_coverage": True,
        "code_sha256": sha256(Path(__file__)),
        "dataset_manifests": {"SIIM": {
            "path": "SIIM/manifest.json",
            "sha256": hashlib.sha256(artifacts["SIIM/manifest.json"]).hexdigest(),
            "fingerprint": manifest["fingerprint"], "support_sha256": support_sha,
            "support_images": len(support), "test_images": len(test),
            "protocol": protocol,
        }},
    }
    root_manifest = {**root_payload, "fingerprint": _fingerprint(root_payload)}
    artifacts["manifest.json"] = _json_bytes(root_manifest)
    output_dir = Path(output_dir).expanduser().resolve()
    _check_artifacts(output_dir, artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_artifacts(output_dir, artifacts)
        for relative, content in artifacts.items():
            destination = output_dir / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".prepare-", dir=destination.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
            temporary.replace(destination)
        _check_artifacts(output_dir, artifacts, require_all=True)
    return root_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.output, args.shots_per_class, args.seed)
    support = load_spec(args.output, "SIIM", "support")
    test = load_spec(args.output, "SIIM", "test")
    for key in ("groups", "ids", "image_paths"):
        if set(support[key]) & set(test[key]):
            raise AssertionError(f"SIIM support/test overlap in {key}")
    print(_json_bytes({
        "status": "passed", "output": str(args.output.resolve()),
        "dataset": manifest["dataset_manifests"]["SIIM"],
    }).decode(), flush=True)


if __name__ == "__main__":
    main()
