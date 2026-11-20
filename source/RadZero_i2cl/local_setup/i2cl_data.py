#!/usr/bin/env python3
"""Immutable few-shot support/evaluation splits for the five public datasets.

The same support.csv is used to extract context vectors and calibrate I2CL
coefficients. Multilabel tasks allocate distinct images to five-example class
quotas, retaining their complete original multilabel targets. Binary tasks use
five positive and five negative examples. Only SIIM has an independent local
training pool; the other four tasks hold support out of the released test pool.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pydicom

from batch_data import DATASETS, load_dataset
from infer import ASSETS_DIR


DEFAULT_OUTPUT = ASSETS_DIR / "splits" / "public5_i2cl_shot5_seed42_v1"
DATA_ROOT = (ASSETS_DIR / "data").resolve()
FIELDS = (
    "index", "image_path", "image_id", "group", "patient_id", "study_id",
    "byte_sha256", "source_split", "anchor_class", "labels_json",
)
PROTOCOL_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        writer.writerow({
            **{key: row.get(key, "") for key in FIELDS},
            "index": index,
            "labels_json": json.dumps(row["labels"], separators=(",", ":")),
        })
    return handle.getvalue().encode("utf-8")


def _source_records(paths):
    return [{"path": str(path), "sha256": sha256(path)}
            for path in sorted({Path(p).resolve() for p in paths}, key=str)]


class _UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def _clean_siim_rows(split):
    directory = ASSETS_DIR / "splits" / "siim_adapter_v1"
    manifest_path = directory / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / f"{split}.csv"
    if sha256(path) != manifest["csv_sha256"][path.name]:
        raise ValueError(f"Existing clean SIIM split checksum mismatch: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            rows.append({
                "image_path": item["image_path"],
                "image_id": item["image_id"],
                "patient_id": item["patient_id"],
                "study_id": item["study_id"],
                "source_split": f"clean_{split}",
                "labels": [int(item["label"])],
            })
    if len(rows) != manifest["counts"][split]["images"]:
        raise ValueError(f"Existing clean SIIM split count mismatch: {path}")
    return rows, [path, manifest_path]


def _inventory(name):
    spec = load_dataset(name, DATA_ROOT)
    source_paths = [Path(path) for path in spec["source_files"]]
    if name == "SIIM":
        train, train_sources = _clean_siim_rows("train")
        test, test_sources = _clean_siim_rows("test")
        if ([row["image_path"] for row in test] != spec["image_paths"] or
                [row["labels"] for row in test] != spec["labels"].tolist()):
            raise ValueError("Clean SIIM test no longer equals released test")
        rows = train + test
        source_paths.extend(train_sources + test_sources)
        support_candidates = list(range(len(train)))
        evaluation_candidates = list(range(len(train), len(rows)))
        grouping = "DICOM PatientID and StudyInstanceUID from checked clean SIIM splits"
        limits = ["Byte-identical image files are grouped; visually near-duplicate pixels are not searched."]
    else:
        rows = [{
            "image_path": path,
            "image_id": Path(path).stem,
            "labels": label.tolist(),
            "patient_id": "",
            "study_id": "",
            "source_split": "released_test",
        } for path, label in zip(spec["image_paths"], spec["labels"])]
        support_candidates = list(range(len(rows)))
        evaluation_candidates = list(range(len(rows)))
        limits = [
            "No separate local training pool: support is held out of the released evaluation pool.",
            "Results are a new few-shot holdout protocol, not the original full-test zero-shot score.",
            "Byte-identical image files are grouped; visually near-duplicate pixels are not searched.",
        ]
        if name == "ChestXray14":
            for row in rows:
                patient = Path(row["image_path"]).stem.split("_")[0]
                if not patient.isdigit() or len(patient) != 8:
                    raise ValueError(f"Unexpected NIH patient filename: {row['image_path']}")
                row["patient_id"] = patient
            grouping = "NIH patient ID from the eight-digit filename prefix"
        elif name == "OpenI":
            for row, study in zip(rows, spec["metadata"]["study_ids"]):
                if not str(study).strip():
                    raise ValueError("OpenI has an empty study uid")
                row["study_id"] = str(study)
            grouping = "OpenI report/study uid from the released custom.csv"
            limits.append("Study IDs are available; patient-level identity across different studies is not supplied.")
        elif name == "RSNA":
            for row in rows:
                header = pydicom.dcmread(
                    DATA_ROOT / row["image_path"], stop_before_pixels=True,
                    specific_tags=["PatientID", "StudyInstanceUID", "SOPInstanceUID"],
                )
                row["patient_id"] = str(getattr(header, "PatientID", "")).strip()
                row["study_id"] = str(getattr(header, "StudyInstanceUID", "")).strip()
                if not row["patient_id"] or not row["study_id"]:
                    raise ValueError(f"RSNA patient/study identity absent: {row['image_path']}")
                if str(getattr(header, "SOPInstanceUID", "")) != row["image_id"]:
                    raise ValueError(f"RSNA SOP ID disagrees with filename: {row['image_path']}")
            grouping = "PatientID and StudyInstanceUID read from all source DICOM headers"
        else:
            grouping = "File SHA256 only; no patient or study identifiers are available"
            limits.append("ChestXDet10 has no patient/study metadata; patient-level separation cannot be guaranteed.")

    paths = [row["image_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name}: duplicate inventory image path")
    for row in rows:
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts or not (DATA_ROOT / relative).is_file():
            raise ValueError(f"{name}: invalid local image path: {relative}")
    print(f"{name}: hashing {len(rows)} image files with 8 CPU workers", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(sha256, [DATA_ROOT / path for path in paths]))
    union = _UnionFind(len(rows))
    representatives = {}
    for index, (row, digest) in enumerate(zip(rows, hashes)):
        row["byte_sha256"] = digest
        for kind in ("patient_id", "study_id", "byte_sha256"):
            if row[kind]:
                key = (kind, row[kind])
                if key in representatives:
                    union.union(index, representatives[key])
                else:
                    representatives[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(union.find(index), []).append(index)
    for members in groups.values():
        group_id = f"{name}:" + _fingerprint(sorted(paths[index] for index in members))[:24]
        for index in members:
            rows[index]["group"] = group_id
    source_paths.extend([
        Path(__file__).resolve(),
        Path(__file__).with_name("batch_data.py").resolve(),
        Path(__file__).with_name("siim_adapter_data.py").resolve(),
    ])
    return spec, rows, support_candidates, evaluation_candidates, {
        "grouping": grouping,
        "limitations": limits,
        "source_files": _source_records(source_paths),
        "image_content_fingerprint": _fingerprint(list(zip(paths, hashes))),
    }


def _select_support(rows, support_candidates, evaluation_candidates, class_names,
                    shots_per_class, seed, independent_train):
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int8)
    num_classes = len(class_names)
    by_group = {}
    for index in evaluation_candidates:
        by_group.setdefault(rows[index]["group"], []).append(index)
    evaluation_labels = labels[evaluation_candidates]
    eval_positive = evaluation_labels.sum(axis=0)
    eval_negative = len(evaluation_candidates) - eval_positive
    if np.any(eval_positive == 0) or np.any(eval_negative == 0):
        raise ValueError("Initial evaluation pool must contain both outcomes for every class")
    if independent_train:
        evaluation_groups = set(by_group)
        support_candidates = [
            index for index in support_candidates if rows[index]["group"] not in evaluation_groups
        ]
    if num_classes == 1:
        anchors = [
            (f"{class_names[0]}:negative", np.asarray([labels[i, 0] == 0 for i in support_candidates])),
            (f"{class_names[0]}:positive", np.asarray([labels[i, 0] == 1 for i in support_candidates])),
        ]
    else:
        anchors = [
            (name, labels[support_candidates, column] == 1)
            for column, name in enumerate(class_names)
        ]
    anchor_candidates = {
        name: np.asarray(support_candidates, dtype=np.int64)[mask].tolist()
        for name, mask in anchors
    }
    if any(len(candidates) < shots_per_class for candidates in anchor_candidates.values()):
        raise ValueError(f"Too few images for the requested class quotas: "
                         f"{ {name: len(items) for name, items in anchor_candidates.items()} }")
    ordered_anchors = sorted(anchor_candidates, key=lambda name: (len(anchor_candidates[name]), name))
    group_positive = {
        group: labels[indices].sum(axis=0) for group, indices in by_group.items()
    }
    group_negative = {
        group: len(indices) - group_positive[group] for group, indices in by_group.items()
    }
    last_error = ""
    for attempt in range(512):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        chosen, assignments, blocked_groups = [], {}, set()
        remaining_positive, remaining_negative = eval_positive.copy(), eval_negative.copy()
        failed = False
        for anchor in ordered_anchors:
            candidates = rng.permutation(anchor_candidates[anchor]).tolist()
            allocated = 0
            for index in candidates:
                if index in assignments:
                    continue
                group = rows[index]["group"]
                add_group = group in by_group and group not in blocked_groups
                if add_group:
                    positive = remaining_positive - group_positive[group]
                    negative = remaining_negative - group_negative[group]
                    if np.any(positive < 1) or np.any(negative < 1):
                        continue
                chosen.append(index)
                assignments[index] = anchor
                allocated += 1
                if add_group:
                    remaining_positive, remaining_negative = positive, negative
                    blocked_groups.add(group)
                if allocated == shots_per_class:
                    break
            if allocated != shots_per_class:
                last_error = f"anchor {anchor}: allocated {allocated}/{shots_per_class}"
                failed = True
                break
        if failed:
            continue
        support_labels = labels[chosen]
        if np.any(support_labels.sum(axis=0) == 0) or np.any(
                support_labels.sum(axis=0) == len(chosen)):
            last_error = "Support set lacks both outcomes for a class"
            continue
        support_groups = {rows[index]["group"] for index in chosen}
        test = [index for index in evaluation_candidates if rows[index]["group"] not in support_groups]
        excluded = [
            index for index in evaluation_candidates
            if rows[index]["group"] in support_groups and index not in assignments
        ]
        if set(chosen) & set(test):
            raise AssertionError("Support/evaluation image intersection")
        if support_groups & {rows[index]["group"] for index in test}:
            raise AssertionError("Support/evaluation group intersection")
        if independent_train and test != evaluation_candidates:
            raise AssertionError("Independent training selection changed the original SIIM test")
        if len(chosen) != shots_per_class * len(anchors):
            raise AssertionError("Wrong support quota total")
        return chosen, test, excluded, assignments, {
            "deterministic_attempt": attempt,
            "anchor_order": ordered_anchors,
            "anchor_counts": {anchor: sum(value == anchor for value in assignments.values())
                              for anchor in anchor_candidates},
            "anchor_class_count": len(anchors),
            "eligible_support_candidates": len(support_candidates),
        }
    raise ValueError(f"No valid support assignment after 512 deterministic attempts: {last_error}")


def _counts(rows, class_names):
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int8)
    positive = labels.sum(axis=0).tolist() if len(rows) else [0] * len(class_names)
    return {
        "images": len(rows),
        "groups": len({row["group"] for row in rows}),
        "known_patients": len({row["patient_id"] for row in rows if row["patient_id"]}),
        "known_studies": len({row["study_id"] for row in rows if row["study_id"]}),
        "positive_images_per_class": dict(zip(class_names, positive)),
        "negative_images_per_class": dict(zip(class_names, [len(rows) - n for n in positive])),
    }


def _build_dataset(name, shots_per_class, seed):
    spec, inventory, candidates, evaluation, provenance = _inventory(name)
    support, test, excluded, anchors, selection = _select_support(
        inventory, candidates, evaluation, spec["class_names"], shots_per_class, seed,
        independent_train=name == "SIIM",
    )
    for index, anchor in anchors.items():
        inventory[index]["anchor_class"] = anchor
    splits = {
        "support": [inventory[index] for index in support],
        "test": [inventory[index] for index in test],
        "excluded": [inventory[index] for index in excluded],
        "inventory": inventory,
    }
    artifacts = {f"{split}.csv": _csv_bytes(rows) for split, rows in splits.items()}
    csv_hashes = {filename: hashlib.sha256(content).hexdigest()
                  for filename, content in artifacts.items()}
    support_sha = csv_hashes["support.csv"]
    counts = {split: _counts(rows, spec["class_names"]) for split, rows in splits.items()}
    payload = {
        "schema_version": PROTOCOL_VERSION,
        "name": name,
        "data_root": str(DATA_ROOT),
        "class_names": spec["class_names"],
        "prompts": spec["prompts"],
        "seed": seed,
        "shots_per_class": shots_per_class,
        "protocol": "clean_train_support_original_test" if name == "SIIM" else "released_test_fewshot_group_holdout",
        "evaluation_unit": "image",
        "context_and_calibration_use_identical_support": True,
        "support_sha256": support_sha,
        "context_support_sha256": support_sha,
        "calibration_support_sha256": support_sha,
        "support_semantics": (
            "Five distinct images per positive and negative state (or requested shots_per_class)."
            if len(spec["class_names"]) == 1 else
            "Distinct image assignments per positive class quota; complete original multilabel targets "
            "are retained, so actual per-class positive counts may exceed the quota."
        ),
        "selection": selection,
        "counts": counts,
        "csv_sha256": csv_hashes,
        "support_evaluation_intersections": {
            key: len({row[key] for row in splits["support"] if row[key]} &
                     {row[key] for row in splits["test"] if row[key]})
            for key in ("image_path", "group", "patient_id", "study_id", "byte_sha256")
        },
        "removed_from_released_evaluation": len(evaluation) - len(test),
        "discarded_non_support_same_group_images": len(excluded),
        "support_rows_in_released_evaluation": len(set(support) & set(evaluation)),
        "evaluation_order": "Released image order with support-associated groups excluded; unchanged for SIIM.",
        "label_semantics": spec["metadata"]["label_source"],
        "source_metadata": spec["metadata"],
        **provenance,
    }
    if any(payload["support_evaluation_intersections"].values()):
        raise AssertionError("Support/evaluation separation audit failed")
    manifest = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts["manifest.json"] = _json_bytes(manifest)
    print(f"{name}: support={len(support)}, evaluation={len(test)}, "
          f"excluded_related={len(excluded)}, attempt={selection['deterministic_attempt']}", flush=True)
    return manifest, artifacts


def _check_artifacts(output_dir, artifacts, require_all=False):
    if not output_dir.exists():
        if require_all:
            raise FileNotFoundError(output_dir)
        return
    for relative, content in artifacts.items():
        destination = output_dir / relative
        if destination.is_symlink():
            raise FileExistsError(f"Refusing to replace an artifact symlink: {destination}")
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                raise FileExistsError(f"Existing split differs; choose a fresh output directory: {destination}")
        elif require_all:
            raise FileNotFoundError(destination)
    allowed = {str(path) for path in artifacts} | {".prepare.lock"}
    unexpected = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*")
                  if path.is_file() and str(path.relative_to(output_dir)) not in allowed]
    if unexpected:
        raise FileExistsError(f"Unexpected files in split output: {unexpected[:5]}")


def prepare_splits(output_dir=DEFAULT_OUTPUT, shots_per_class=5, seed=42, datasets=None):
    """Build immutable, deterministic support/test splits and return root manifest."""
    if not isinstance(shots_per_class, int) or shots_per_class < 1:
        raise ValueError("shots_per_class must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    datasets = list(DATASETS) if datasets is None else list(datasets)
    if not datasets or len(datasets) != len(set(datasets)) or set(datasets) - set(DATASETS):
        raise ValueError(f"datasets must be a unique nonempty subset of {DATASETS}")
    datasets = [name for name in DATASETS if name in datasets]
    output_dir = Path(output_dir).expanduser().resolve()
    artifacts, manifests = {}, {}
    for name in datasets:
        manifest, items = _build_dataset(name, shots_per_class, seed)
        manifests[name] = manifest
        artifacts.update({f"{name}/{filename}": content for filename, content in items.items()})
    payload = {
        "schema_version": PROTOCOL_VERSION,
        "datasets": datasets,
        "data_root": str(DATA_ROOT),
        "seed": seed,
        "shots_per_class": shots_per_class,
        "context_and_calibration_use_identical_support": True,
        "independent_model_per_dataset": True,
        "validation_or_test_used_for_coefficient_selection": False,
        "split_selection_uses_labels_only_for_class_quota_and_outcome_coverage": True,
        "code_sha256": sha256(Path(__file__)),
        "dataset_manifests": {
            name: {
                "path": f"{name}/manifest.json",
                "sha256": hashlib.sha256(artifacts[f"{name}/manifest.json"]).hexdigest(),
                "fingerprint": manifest["fingerprint"],
                "support_sha256": manifest["support_sha256"],
                "support_images": manifest["counts"]["support"]["images"],
                "test_images": manifest["counts"]["test"]["images"],
                "protocol": manifest["protocol"],
            } for name, manifest in manifests.items()
        },
    }
    root_manifest = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts["manifest.json"] = _json_bytes(root_manifest)
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
            with tempfile.NamedTemporaryFile(prefix=".prepare-", dir=destination.parent,
                                             delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
            temporary.replace(destination)
        _check_artifacts(output_dir, artifacts, require_all=True)
    return root_manifest


def load_spec(split_dir, name, split="support"):
    """Load a verified immutable support/test spec compatible with batch_data."""
    if split not in ("support", "test"):
        raise ValueError("split must be 'support' or 'test'")
    directory = Path(split_dir).expanduser().resolve()
    root_path = directory / "manifest.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    if _fingerprint({key: value for key, value in root.items() if key != "fingerprint"}) != root["fingerprint"]:
        raise ValueError(f"Root manifest fingerprint mismatch: {root_path}")
    if name not in root["datasets"]:
        raise ValueError(f"{name} is absent from {directory}")
    dataset_path = directory / name / "manifest.json"
    if sha256(dataset_path) != root["dataset_manifests"][name]["sha256"]:
        raise ValueError(f"Dataset manifest checksum mismatch: {dataset_path}")
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if _fingerprint({key: value for key, value in manifest.items() if key != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError(f"Dataset manifest fingerprint mismatch: {dataset_path}")
    table = dataset_path.parent / f"{split}.csv"
    if sha256(table) != manifest["csv_sha256"][table.name]:
        raise ValueError(f"Split CSV checksum mismatch: {table}")
    support_path = dataset_path.parent / "support.csv"
    if sha256(support_path) != manifest["support_sha256"]:
        raise ValueError(f"Support CSV checksum mismatch: {support_path}")
    rows = []
    with table.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if int(row["index"]) != index:
                raise ValueError(f"Nonconsecutive split index: {table}")
            path = Path(row["image_path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe split image path: {path}")
            if not (Path(manifest["data_root"]) / path).is_file():
                raise FileNotFoundError(path)
            row["labels"] = json.loads(row["labels_json"])
            rows.append(row)
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int8)
    if labels.shape != (manifest["counts"][split]["images"], len(manifest["class_names"])):
        raise ValueError(f"Split label shape mismatch: {table}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Nonbinary split labels: {table}")
    positive = labels.sum(axis=0)
    if np.any(positive == 0) or np.any(positive == len(rows)):
        raise ValueError(f"Split must have positive and negative examples for every class: {table}")
    if len({row["image_path"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate split images: {table}")
    metadata = {
        **manifest["source_metadata"],
        "split": split,
        "data_root": manifest["data_root"],
        "protocol": manifest["protocol"],
        "unique_images": len(rows),
        "positive_images_per_class": dict(zip(manifest["class_names"], positive.tolist())),
        "split_manifest": str(dataset_path),
        "split_fingerprint": manifest["fingerprint"],
        "root_split_fingerprint": root["fingerprint"],
        "support_sha256": manifest["support_sha256"],
        "context_support_sha256": manifest["support_sha256"],
        "calibration_support_sha256": manifest["support_sha256"],
        "same_support_for_context_and_calibration": True,
        "grouping": manifest["grouping"],
        "limitations": manifest["limitations"],
        "anchor_classes": [row["anchor_class"] for row in rows],
        "byte_sha256": [row["byte_sha256"] for row in rows],
        "patient_ids": [row["patient_id"] for row in rows],
        "study_ids": [row["study_id"] for row in rows],
        "image_ids": [row["image_id"] for row in rows],
        "source_rows": len(rows),
    }
    return {
        "name": name,
        "data_root": manifest["data_root"],
        "image_paths": [row["image_path"] for row in rows],
        "labels": labels,
        "class_names": manifest["class_names"],
        "prompts": manifest["prompts"],
        "ids": [row["image_id"] for row in rows],
        "groups": [row["group"] for row in rows],
        "source_files": sorted({
            str(root_path), str(dataset_path), str(table), str(support_path),
            *(item["path"] for item in manifest["source_files"]),
        }),
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    args = parser.parse_args()
    manifest = prepare_splits(args.output, args.shots_per_class, args.seed, args.datasets)
    for name in manifest["datasets"]:
        support = load_spec(args.output, name, "support")
        test = load_spec(args.output, name, "test")
        if set(support["groups"]) & set(test["groups"]):
            raise AssertionError(f"{name}: group intersection after reload")
    print(json.dumps({"status": "passed", "output": str(args.output.resolve()),
                      "datasets": manifest["dataset_manifests"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
