#!/usr/bin/env python3
"""Reduce existing multilabel supports while preserving the exact prior test.

Choose one image within each parent support anchor quota, using only support
labels for outcome coverage. Unselected parent supports and every previously
excluded related image remain outside evaluation. This is a nested support
experiment, not a fresh split of the released evaluation pool.
"""

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from i2cl_data import (
    PROTOCOL_VERSION, _check_artifacts, _counts, _csv_bytes, _fingerprint,
    _json_bytes, _source_records, load_spec, sha256,
)
from infer import ASSETS_DIR


DATASETS = ("ChestXDet10", "ChestXray14", "OpenI")
DEFAULT_PARENT = ASSETS_DIR / "splits/native_public5_local_seed42"
DEFAULT_OUTPUT = ASSETS_DIR / "splits/native_multilabel_shot1_seed42_fixedtest"
PROTOCOL = "support_subset_fixed_test"
IDENTITY_COLUMNS = (
    "image_path", "image_id", "group", "patient_id", "study_id", "byte_sha256",
)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        if int(row["index"]) != index:
            raise ValueError(f"Nonconsecutive row index in {path}")
        row["labels"] = json.loads(row["labels_json"])
    return rows


def select_subset(rows, class_names, anchor_order, seed):
    """Select from parent supports only; never receive test labels or scores."""
    if set(anchor_order) != set(class_names) or len(anchor_order) != len(class_names):
        raise ValueError("Parent anchor order does not contain exactly all classes")
    candidates = {
        anchor: [i for i, row in enumerate(rows) if row["anchor_class"] == anchor]
        for anchor in anchor_order
    }
    if any(not items for items in candidates.values()):
        raise ValueError("Parent support lacks an anchor class")
    for anchor, indices in candidates.items():
        column = class_names.index(anchor)
        if any(rows[index]["labels"][column] != 1 for index in indices):
            raise ValueError(f"Parent assigned a negative image to anchor {anchor}")
    for attempt in range(512):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        selected = sorted(
            int(rng.permutation(candidates[anchor])[0]) for anchor in anchor_order
        )
        labels = np.asarray([rows[index]["labels"] for index in selected], dtype=np.int8)
        positive = labels.sum(axis=0)
        if np.all(positive > 0) and np.all(positive < len(selected)):
            return selected, {
                "method": "one_per_parent_anchor_class",
                "seed": seed,
                "deterministic_attempt": attempt,
                "anchor_order": anchor_order,
                "anchor_counts": {anchor: 1 for anchor in class_names},
                "anchor_class_count": len(class_names),
                "eligible_support_candidates": len(rows),
                "parent_support_indices": selected,
                "uses_test_labels": False,
                "uses_model_scores": False,
                "retry_condition": "Support must contain both outcomes for every class.",
            }
    raise ValueError("No support-only outcome-complete subset in 512 deterministic attempts")


def build_dataset(parent, name, seed):
    # Validate native manifests, labels and paths before copying any artifact.
    load_spec(parent, name, "support")
    load_spec(parent, name, "test")
    parent_dir = parent / name
    parent_manifest_path = parent_dir / "manifest.json"
    original = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    source_rows, source_bytes = {}, {}
    for split in ("support", "test", "excluded", "inventory"):
        path = parent_dir / f"{split}.csv"
        if sha256(path) != original["csv_sha256"][path.name]:
            raise ValueError(f"Parent split checksum mismatch: {path}")
        source_rows[split] = read_rows(path)
        source_bytes[split] = path.read_bytes()
    for record in original["source_files"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Parent original source changed: {record['path']}")
    parent_support = source_rows["support"]
    classes = original["class_names"]
    anchor_counts = Counter(row["anchor_class"] for row in parent_support)
    if anchor_counts != {name: original["shots_per_class"] for name in classes}:
        raise ValueError("Parent support anchor quotas are inconsistent")
    selected, selection = select_subset(
        parent_support, classes, original["selection"]["anchor_order"], seed,
    )
    support = [parent_support[index] for index in selected]
    support_paths = {row["image_path"] for row in support}
    unused = [row for row in parent_support if row["image_path"] not in support_paths]
    test = source_rows["test"]
    test_paths = {row["image_path"] for row in test}
    inventory = source_rows["inventory"]
    inventory_paths = {row["image_path"] for row in inventory}
    parent_excluded_paths = {row["image_path"] for row in source_rows["excluded"]}
    parent_support_paths = {row["image_path"] for row in parent_support}
    if (parent_support_paths | test_paths | parent_excluded_paths) != inventory_paths:
        raise ValueError("Parent inventory is not its support/test/excluded partition")
    if (parent_support_paths & test_paths or parent_support_paths & parent_excluded_paths
            or test_paths & parent_excluded_paths):
        raise ValueError("Parent partition overlaps")
    excluded = [
        row for row in inventory
        if row["image_path"] not in support_paths and row["image_path"] not in test_paths
    ]
    lookup = {row["image_path"]: row for row in inventory}
    for row in support + test:
        expected = lookup[row["image_path"]]
        for key in (*IDENTITY_COLUMNS, "anchor_class", "labels", "source_split"):
            if row[key] != expected[key]:
                raise ValueError(f"{name}: split/inventory mismatch in {key}")
    intersections = {
        key: len({row[key] for row in support if row[key]} &
                 {row[key] for row in test if row[key]})
        for key in IDENTITY_COLUMNS
    }
    if any(intersections.values()):
        raise ValueError(f"Support/test identity overlap: {intersections}")
    parent_group_overlap = len(
        {row["group"] for row in parent_support} & {row["group"] for row in test}
    )
    if parent_group_overlap:
        raise ValueError("Fixed test includes a parent support group")
    # Only the selected supports are newly read/hashes checked here. Full image
    # inventory hashes and grouping remain the independently checked parent record.
    for row in support:
        if sha256(Path(original["data_root"]) / row["image_path"]) != row["byte_sha256"]:
            raise ValueError(f"Selected support image changed: {row['image_path']}")
    splits = {"support": support, "test": test, "excluded": excluded, "inventory": inventory}
    artifacts = {
        "support.csv": _csv_bytes(support),
        "test.csv": source_bytes["test"],
        "excluded.csv": _csv_bytes(excluded),
        "inventory.csv": source_bytes["inventory"],
    }
    hashes = {filename: hashlib.sha256(content).hexdigest()
              for filename, content in artifacts.items()}
    positive = np.asarray([row["labels"] for row in support], dtype=np.float32).sum(axis=0)
    negative = len(support) - positive
    support_groups = {row["group"] for row in support}
    source_records = _source_records([
        *(record["path"] for record in original["source_files"]),
        parent / "manifest.json", parent_manifest_path, Path(__file__),
        *(parent_dir / f"{split}.csv" for split in source_rows),
    ])
    payload = {
        **{key: original[key] for key in (
            "name", "data_root", "class_names", "prompts", "evaluation_unit",
            "label_semantics", "source_metadata", "grouping", "image_content_fingerprint",
        )},
        "schema_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL,
        "seed": seed,
        "shots_per_class": 1,
        "parent_split": {
            "path": str(parent),
            "manifest_path": str(parent_manifest_path),
            "manifest_sha256": sha256(parent_manifest_path),
            "root_manifest_sha256": sha256(parent / "manifest.json"),
            "fingerprint": original["fingerprint"],
            **{f"{split}_sha256": original["csv_sha256"][f"{split}.csv"]
               for split in source_rows},
        },
        "context_and_calibration_use_identical_support": True,
        "support_sha256": hashes["support.csv"],
        "context_support_sha256": hashes["support.csv"],
        "calibration_support_sha256": hashes["support.csv"],
        "support_semantics": (
            "One distinct image per parent anchor_class, sampled only from its "
            "parent support quota. Complete original multilabel targets retained. "
            "Actual positive counts can exceed one per disease."
        ),
        "selection": selection,
        "counts": {split: _counts(items, classes) for split, items in splits.items()},
        "csv_sha256": hashes,
        "support_evaluation_intersections": intersections,
        "parent_support_group_test_intersections": parent_group_overlap,
        "removed_from_released_evaluation": len(inventory) - len(test),
        "support_rows_in_released_evaluation": len(support),
        "discarded_non_support_same_group_images": sum(
            row["group"] in support_groups for row in excluded
        ),
        "unused_parent_support_images": len(unused),
        "unused_parent_support_image_paths": [row["image_path"] for row in unused],
        "parent_excluded_images": len(source_rows["excluded"]),
        "excluded_semantics": (
            "All parent excluded rows plus every unselected parent support, in "
            "inventory order. No removed support or related image returns to test."
        ),
        "evaluation_order": "Byte-identical parent test.csv, including image order and labels.",
        "inventory_semantics": (
            "Byte-identical parent inventory.csv; its anchor_class column records "
            "parent quota assignments, not current membership in support."
        ),
        "support_order": "Parent support order with unselected rows removed.",
        "test_sha256_unchanged": hashes["test.csv"] == original["csv_sha256"]["test.csv"],
        "inventory_sha256_unchanged": (
            hashes["inventory.csv"] == original["csv_sha256"]["inventory.csv"]
        ),
        "support_pos_weight_diagnostic": {
            "positive_counts": positive.astype(np.int64).tolist(),
            "negative_counts": negative.astype(np.int64).tolist(),
            "pos_weight": (negative / positive).tolist(),
            "statistics_split": "selected support only",
        },
        "limitations": [
            *original["limitations"],
            "The test remains fixed to the parent five-shot experiment; unused "
            "parent supports and all parent excluded groups stay excluded.",
            "One seed and a nested support subset are not a repeated-seed confidence estimate.",
        ],
        "source_files": source_records,
    }
    manifest = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts["manifest.json"] = _json_bytes(manifest)
    return manifest, artifacts


def prepare(parent=DEFAULT_PARENT, output=DEFAULT_OUTPUT, seed=42):
    parent = Path(parent).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if seed < 0 or output == parent:
        raise ValueError("Use a nonnegative seed and a fresh output directory")
    artifacts, manifests = {}, {}
    for name in DATASETS:
        manifest, items = build_dataset(parent, name, seed)
        manifests[name] = manifest
        artifacts.update({f"{name}/{filename}": data for filename, data in items.items()})
    payload = {
        "schema_version": PROTOCOL_VERSION,
        "datasets": list(DATASETS),
        "data_root": manifests[DATASETS[0]]["data_root"],
        "seed": seed,
        "shots_per_class": 1,
        "protocol": PROTOCOL,
        "parent_split_root": str(parent),
        "parent_root_manifest_sha256": sha256(parent / "manifest.json"),
        "context_and_calibration_use_identical_support": True,
        "independent_model_per_dataset": True,
        "validation_or_test_used_for_coefficient_selection": False,
        "split_selection_uses_labels_only_for_class_quota_and_outcome_coverage": True,
        "test_labels_used_for_subset_selection": False,
        "code_sha256": sha256(Path(__file__)),
        "dataset_manifests": {
            name: {
                "path": f"{name}/manifest.json",
                "sha256": hashlib.sha256(artifacts[f"{name}/manifest.json"]).hexdigest(),
                "fingerprint": manifest["fingerprint"],
                "support_sha256": manifest["support_sha256"],
                "support_images": manifest["counts"]["support"]["images"],
                "test_images": manifest["counts"]["test"]["images"],
                "protocol": PROTOCOL,
            } for name, manifest in manifests.items()
        },
    }
    root = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts["manifest.json"] = _json_bytes(root)
    _check_artifacts(output, artifacts)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_artifacts(output, artifacts)
        for relative, content in artifacts.items():
            destination = output / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".prepare-", dir=destination.parent, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
            temporary.replace(destination)
        _check_artifacts(output, artifacts, require_all=True)
    for name in DATASETS:
        support, test = [load_spec(output, name, split) for split in ("support", "test")]
        if len(support["image_paths"]) != len(support["class_names"]):
            raise AssertionError(f"{name}: expected one support per disease")
        if set(support["groups"]) & set(test["groups"]):
            raise AssertionError(f"{name}: group overlap after reload")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.parent, args.output, args.seed)
    print(_json_bytes({
        "status": "passed", "output": str(args.output.resolve()),
        "datasets": result["dataset_manifests"],
    }).decode(), flush=True)


if __name__ == "__main__":
    main()
