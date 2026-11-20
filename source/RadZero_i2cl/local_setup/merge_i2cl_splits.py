#!/usr/bin/env python3
"""Combine existing immutable split roots without resampling any image."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

from batch_data import DATASETS
from i2cl_data import (
    _check_artifacts, _fingerprint, _json_bytes, load_spec, sha256,
)


def merge(inputs, output):
    entries, artifacts, roots, common, exceptions = {}, {}, [], None, {}
    for source in inputs:
        source = Path(source).expanduser().resolve()
        root_path = source / "manifest.json"
        root = json.loads(root_path.read_text(encoding="utf-8"))
        config = {key: root[key] for key in
                  ("schema_version", "data_root", "seed", "shots_per_class")}
        if common is not None and config != common:
            raise ValueError("Cannot combine roots with different data/seed/shot settings")
        common = config
        roots.append({"path": str(source), "sha256": sha256(root_path),
                      "fingerprint": root["fingerprint"]})
        for name in root["datasets"]:
            if name in entries:
                raise ValueError(f"Dataset appears in more than one source root: {name}")
            # Native load_spec checks root, dataset and CSV hashes plus every path.
            support = load_spec(source, name, "support")
            test = load_spec(source, name, "test")
            for key in ("image_paths", "ids", "groups"):
                if set(support[key]) & set(test[key]):
                    raise ValueError(f"{name} support/test overlap in {key}")
            entries[name] = root["dataset_manifests"][name]
            directory = source / name
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    artifacts[str(path.relative_to(source))] = path.read_bytes()
            dataset = json.loads(artifacts[f"{name}/manifest.json"])
            if "protocol_exception" in dataset:
                exceptions[name] = dataset["protocol_exception"]
            if hashlib.sha256(artifacts[f"{name}/manifest.json"]).hexdigest() != entries[name]["sha256"]:
                raise ValueError(f"{name} source changed while merging")
    if not entries:
        raise ValueError("At least one existing dataset split is required")
    names = [name for name in DATASETS if name in entries]
    if set(names) != set(entries):
        raise ValueError("Source contains an unknown dataset")
    payload = {
        **common,
        "datasets": names,
        "context_and_calibration_use_identical_support": True,
        "independent_model_per_dataset": True,
        "validation_or_test_used_for_coefficient_selection": False,
        "split_selection_uses_labels_only_for_class_quota_and_outcome_coverage": True,
        "code_sha256": sha256(Path(__file__)),
        "dataset_manifests": {name: entries[name] for name in names},
        "source_split_roots": roots,
        "merge_semantics": (
            "Dataset directories are byte-for-byte copies of validated source "
            "splits. No images, labels, groups, ordering or selections changed."
        ),
        "protocol_exceptions": exceptions,
    }
    root_manifest = {**payload, "fingerprint": _fingerprint(payload)}
    artifacts["manifest.json"] = _json_bytes(root_manifest)
    output = Path(output).expanduser().resolve()
    if output in [Path(value).expanduser().resolve() for value in inputs]:
        raise ValueError("Merged output must differ from every source directory")
    _check_artifacts(output, artifacts)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _check_artifacts(output, artifacts)
        for relative, content in artifacts.items():
            path = output / relative
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".merge-", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
            temporary.replace(path)
        _check_artifacts(output, artifacts, require_all=True)
    for name in names:
        for split in ("support", "test"):
            load_spec(output, name, split)
    return root_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(args.inputs, args.output)
    print(_json_bytes({
        "status": "passed", "output": str(args.output.resolve()),
        "datasets": result["dataset_manifests"],
    }).decode(), flush=True)


if __name__ == "__main__":
    main()
