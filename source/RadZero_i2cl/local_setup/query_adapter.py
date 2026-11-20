#!/usr/bin/env python3
"""Portable frozen-query adapter: original/refined vectors, with no support labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
DEFAULT_ROOT = PROJECT / "runs/overnight_v2_20261020_2250/selected_query_adapter"
FORMAT = "radzero-frozen-query-adapter-v1"
FORBIDDEN_METADATA = {"labels", "support_labels", "support_images", "support_features",
                      "positive_counts", "negative_counts", "positive_means", "negative_means"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class QueryAdapter:
    """Two immutable class query matrices; the large RadZero model stays separate."""

    def __init__(self, original, refined, metadata):
        self.original = np.asarray(original).copy()
        self.refined = np.asarray(refined).copy()
        self.metadata = dict(metadata)
        def validate_metadata(value):
            if isinstance(value, dict):
                if FORBIDDEN_METADATA & set(value):
                    raise ValueError("Support examples, labels and class statistics must not be exported")
                for item in value.values():
                    validate_metadata(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    validate_metadata(item)
        validate_metadata(self.metadata)
        if self.metadata["format"] != FORMAT:
            raise ValueError("Unsupported query adapter format")
        self.eta = float(self.metadata["eta"])
        self.class_names = tuple(self.metadata["class_names"])
        self.prompts = tuple(self.metadata["prompts"])
        if (not np.isfinite(self.eta) or self.eta < 0 or not self.class_names
                or len(set(self.class_names)) != len(self.class_names)
                or len(self.prompts) != len(self.class_names)):
            raise ValueError("Invalid eta, class names, or prompt order")
        for values in (self.original, self.refined):
            if (values.dtype != np.float32 or values.ndim != 2
                    or values.shape[0] != len(self.class_names)
                    or not np.isfinite(values).all()):
                raise ValueError("Expected finite FP32 [class,hidden] query matrices")
        if self.original.shape != self.refined.shape:
            raise ValueError("Original/refined query dimensions differ")
        if self.eta == 0 and not np.array_equal(self.original, self.refined):
            raise ValueError("Eta zero must preserve the exact original query")
        self.original.setflags(write=False)
        self.refined.setflags(write=False)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path / "query_adapter.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        if sha256(path) != metadata["query_file_sha256"]:
            raise ValueError("Query adapter checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"original_queries", "refined_queries"}:
                raise ValueError("Adapter may contain only original/refined query arrays")
            result = cls(arrays["original_queries"], arrays["refined_queries"], metadata)
        result.path = path
        return result

    def make_matcher(self, model):
        from overnight_query_refinement import QueryRefinementMatcher
        if self.eta == 0:
            values, etas = self.original[None].copy(), (0.,)
        else:
            values = np.stack((self.original, self.refined))
            etas = (0., self.eta)
        return QueryRefinementMatcher(model, values, etas)


def save_bundle(path, original, refined, metadata):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing existing adapter: {path}")
    checked = QueryAdapter(original, refined, {**metadata, "format": FORMAT})
    np.savez(path, original_queries=checked.original, refined_queries=checked.refined)
    record = {**metadata, "format": FORMAT, "query_file_sha256": sha256(path),
              "query_shape": list(checked.original.shape), "dtype": "float32",
              "contains_support_images_or_labels": False,
              "serialized_arrays": ["original_queries", "refined_queries"]}
    write_json(path.with_suffix(".json"), record)
    loaded = QueryAdapter.load(path)
    np.testing.assert_array_equal(loaded.original, original)
    np.testing.assert_array_equal(loaded.refined, refined)
    return loaded


def export_dataset(query_run, destination, eta=.3):
    """Export already-fitted vectors; do not open any support labels/images."""
    query_run, destination = Path(query_run).resolve(), Path(destination).resolve()
    manifest_path = query_run / "manifest.json"
    source = json.loads(manifest_path.read_text())
    if source["status"] != "complete":
        raise ValueError("Query run is not complete")
    fit_path = query_run / "fit_vectors.npz"
    if sha256(fit_path) != source["fit_vectors_sha256"]:
        raise ValueError("Query-fit provenance mismatch")
    cache_path = Path(source["cache"]) / "manifest.json"
    if sha256(cache_path) != source["cache_manifest_sha256"]:
        raise ValueError("Frozen cache manifest changed")
    cache = json.loads(cache_path.read_text())
    if cache["status"] != "complete" or cache["checkpoint"] != source["checkpoint"]:
        raise ValueError("Incomplete cache or inconsistent base model")
    for key in ("class_names", "prompts", "support_identity", "test_identity"):
        if source[key] != cache[key]:
            raise ValueError(f"Query/cache {key} mismatch")
    base_sources = dict(cache["source_sha256"])
    for path, digest in source["source_sha256"].items():
        if path in base_sources and base_sources[path] != digest:
            raise ValueError(f"Query/cache source identity differs: {path}")
        base_sources[path] = digest
    for path, digest in {**base_sources, **source["split_sha256"]}.items():
        if not Path(path).is_file() or sha256(path) != digest:
            raise ValueError(f"Recorded source/split changed: {path}")
    # Access only the three named arrays. np.load does not deserialize the
    # support statistics/labels that happen to exist in the source archive.
    with np.load(fit_path, allow_pickle=False) as fit:
        etas = np.asarray(fit["etas"], dtype=float)
        hits = np.flatnonzero(etas == float(eta))
        if len(hits) != 1 or np.count_nonzero(etas == 0.) != 1:
            raise ValueError("Requested eta/original query not uniquely available")
        original = np.asarray(fit["original_queries"]).copy()
        refined = np.asarray(fit["queries"][int(hits[0])]).copy()
        np.testing.assert_array_equal(fit["queries"][int(np.flatnonzero(etas == 0.)[0])], original)
    text_path = cache_path.parent / "text_features.npy"
    if sha256(text_path) != cache["auxiliary_files"]["text_features.npy"]["sha256"]:
        raise ValueError("Original text features changed")
    np.testing.assert_array_equal(original, np.load(text_path, allow_pickle=False))
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "dataset": source["dataset"], "eta": float(eta),
        "class_names": source["class_names"], "prompts": source["prompts"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": source["checkpoint"], "trainable_base_parameters": 0,
        "model_config_sha256": cache["model_config_sha256"],
        "preprocessor_sha256": cache["preprocessor_sha256"],
        "base_source_sha256": base_sources,
        "export_source_sha256": {
            str(Path(__file__).with_name(name).resolve()):
                sha256(Path(__file__).with_name(name))
            for name in ("query_adapter.py", "eval_query_adapter.py", "python_local.sh", "activate.sh")
        },
        "cache_manifest_sha256": source["cache_manifest_sha256"],
        "cache_identity": source["cache_identity"],
        "support_identity": source["support_identity"], "test_identity": source["test_identity"],
        "split_dir": source["split_dir"], "split_sha256": source["split_sha256"],
        "source_query_run": str(query_run), "source_query_manifest_sha256": sha256(manifest_path),
        "source_fit_vectors_sha256": source["fit_vectors_sha256"],
        "reference_batch_size": source["batch_size"],
        "method": "Original frozen RadZero cross-attention with a support-derived refined class query",
        "map_semantics": "Both logits and similarity maps use the corresponding original/refined query.",
        "selection": "Eta was selected by exploratory comparison on the reused test sets; not support-OOF-selected.",
        "fit_split": "support_only", "approval_state": "pending_final_root_audit",
        "portability": "Only two query matrices are required. Keep the matching frozen RadZero checkpoint and scorer code; no support labels/features are needed for inference.",
    }
    adapter = save_bundle(destination / "query_adapter.npz", original, refined, metadata)
    write_json(destination / "EXPORT_AUDIT.json", {
        "status": "passed", "queries_bitwise_equal_to_completed_run": True,
        "original_query_bitwise_equal_to_frozen_text_features": True,
        "support_labels_or_images_exported": False,
        "adapter_sha256": sha256(adapter.path), "source_fit_sha256": sha256(fit_path),
    })
    return adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--eta", type=float, default=.3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = {}
    for dataset in args.datasets:
        adapter = export_dataset(args.query_root / dataset, args.output / dataset, args.eta)
        records[dataset] = {"path": str(adapter.path), "sha256": sha256(adapter.path),
                            "classes": len(adapter.class_names), "eta": adapter.eta}
    write_json(args.output / "manifest.json", {
        "status": "exported_pending_final_audit", "datasets": records,
        "export_source_sha256": sha256(__file__), "encoder_execution": False,
        "support_labels_or_images_exported": False, "default_eta": args.eta,
    })
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
