#!/usr/bin/env python3
"""Compact class-local weighted-ridge score adapter; no support data at inference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

import numpy as np
from threadpoolctl import threadpool_limits

from overnight_cache_methods import normalized_features, sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT / "runs/overnight_20261020_2150"
RAW = PROJECT / "local_assets/outputs/overnight_20261020_2150"
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
DEFAULT_ADAPTER_ROOT = CAMPAIGN / "selected_adapter"
FORMAT = "radzero-class-local-weighted-ridge-adapter-v1"
NPZ_KEYS = {"directions", "strength", "class_names", "prompts", "metadata_json"}


class RidgeAdapter:
    def __init__(self, directions, strength, class_names, prompts, metadata):
        self.directions = np.array(directions, dtype=np.float64, copy=True, order="C")
        self.strength = float(strength)
        self.class_names = tuple(class_names)
        self.prompts = tuple(prompts)
        self.metadata = dict(metadata)
        if (self.directions.shape != (len(self.class_names), 768)
                or len(self.prompts) != len(self.class_names) or not self.class_names
                or len(set(self.class_names)) != len(self.class_names)):
            raise ValueError("Expected unique classes/prompts and directions [C,768]")
        if (not np.isfinite(self.directions).all()
                or not math.isfinite(self.strength) or self.strength < 0):
            raise ValueError("Nonfinite/negative adapter parameters")
        if any(not isinstance(value, str) or not value for value in self.class_names + self.prompts):
            raise ValueError("Class names/prompts must be nonempty strings")
        self.directions.setflags(write=False)

    def predict(self, base_logits, class_local_features):
        """No labels accepted. Raw baseline + strength * normalized-feature dot W."""
        base = np.asarray(base_logits, dtype=np.float64)
        if base.ndim != 2 or base.shape[1] != len(self.class_names) or not np.isfinite(base).all():
            raise ValueError("Baseline must be finite raw logits [N,C]")
        feature = normalized_features(class_local_features, len(self.class_names))
        if feature.shape != (len(base), len(self.class_names), 768):
            raise ValueError("Expected class-local features [N,C,768]")
        residual = np.empty_like(base)
        # Preserve the original CPU float64 classwise matmul operation.
        with threadpool_limits(limits=4):
            for index in range(len(self.class_names)):
                residual[:, index] = feature[:, index] @ self.directions[index]
        logits = base + self.strength * residual
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite adapted logits")
        return logits

    def save(self, path):
        path = Path(path)
        metadata = dict(self.metadata, format=FORMAT,
                        directions_sha256=hashlib.sha256(self.directions.tobytes()).hexdigest())
        with path.open("xb") as handle:
            np.savez_compressed(
                handle, directions=self.directions, strength=np.float64(self.strength),
                class_names=np.asarray(self.class_names), prompts=np.asarray(self.prompts),
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != NPZ_KEYS:
                raise ValueError("Unexpected archive fields; only compact inference state is allowed")
            if archive["directions"].dtype != np.float64 or archive["strength"].shape != ():
                raise ValueError("Invalid adapter dtype/strength shape")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata.get("format") != FORMAT:
                raise ValueError("Unknown adapter format")
            adapter = cls(archive["directions"], archive["strength"].item(),
                          archive["class_names"].tolist(), archive["prompts"].tolist(), metadata)
        if hashlib.sha256(adapter.directions.tobytes()).hexdigest() != metadata["directions_sha256"]:
            raise ValueError("Adapter direction checksum mismatch")
        return adapter


def find_selected(run_dir):
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    candidates = [record for record in summary["results"]
                  if record["config"]["method"] == "ridge"
                  and record["config"]["feature"] == "class_local"]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one support-selected class_local ridge result")
    return candidates[0]


def export_dataset(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    record = find_selected(run_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    cache_path = Path(config["cache_manifest"])
    cache = json.loads(cache_path.read_text())
    if cache["status"] != "complete" or sha256(cache_path) != config["cache_manifest_sha256"]:
        raise ValueError("Frozen cache identity changed")
    candidate = run_dir / record["candidate_id"]
    if sha256(candidate / "fit.npz") != record["fit_sha256"]:
        raise ValueError("Selected fit checksum changed")
    if sha256(run_dir / "selection.json") != record["selection_sha256"]:
        raise ValueError("Support selection checksum changed")
    if sha256(candidate / "predictions.npy") != record["prediction_sha256"]:
        raise ValueError("Reference predictions changed")
    with np.load(candidate / "fit.npz", allow_pickle=False) as fit:
        if str(fit["method"].item()) != "ridge":
            raise ValueError("Selected fit is not ridge")
        directions = fit["directions"].astype(np.float64, copy=True)
        valid = fit["valid"]
        directions[~valid] = 0.
        penalty = float(fit["parameter"])
        support_sha = str(fit["support_identity"].item())
    metadata = {
        "dataset": cache["dataset"], "method": "class-local weighted ridge least-squares score residual",
        "fit_objective": "Class-balanced ridge fit to y-sigmoid(frozen baseline logits); no BCE fitting",
        "selection": "Support-group OOF mean AUROC/AUPRC; weighted BCE only breaks exact ties",
        "ridge_penalty": penalty, "prediction": "raw baseline logits + strength * float64 normalized class-local dot directions",
        "map_interpretation": "Original RadZero similarity maps are base-model references and do not explain the added score residual",
        "support_data_in_archive": False, "test_labels_in_archive": False,
        "checkpoint": cache["checkpoint"],
        "model_config_sha256": cache["model_config_sha256"],
        "preprocessor_sha256": cache["preprocessor_sha256"],
        "processor_size": {"height": 518, "width": 518},
        "loss_settings": cache["loss_settings"],
        "split_dir": cache["split_dir"], "split_sha256": cache["split_sha256"],
        "root_split_fingerprint": cache["root_split_fingerprint"],
        "support_csv_sha256": support_sha,
        "base_source_sha256": cache["source_sha256"],
        "cache_manifest": str(cache_path), "cache_manifest_sha256": sha256(cache_path),
        "selection_sha256": record["selection_sha256"], "fit_sha256": record["fit_sha256"],
        "reference_predictions_sha256": record["prediction_sha256"],
        "source_candidate": str(candidate), "export_source_sha256": sha256(__file__),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    adapter = RidgeAdapter(directions, record["config"]["strength"],
                           cache["class_names"], cache["prompts"], metadata)
    output.mkdir(parents=True, exist_ok=False)
    adapter_path = output / "adapter.npz"
    adapter.save(adapter_path)
    loaded = RidgeAdapter.load(adapter_path)
    test = cache["splits"]["test"]["files"]
    arrays = {}
    for key in ("class_local", "base_logits"):
        path = cache_path.parent / test[key]["file"]
        if sha256(path) != test[key]["sha256"]:
            raise ValueError("Reference cache array changed")
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = np.load(candidate / "predictions.npy", mmap_mode="r", allow_pickle=False)
    max_error, equal = 0., True
    for start in range(0, len(expected), 512):
        stop = min(start + 512, len(expected))
        actual = loaded.predict(arrays["base_logits"][start:stop], arrays["class_local"][start:stop])
        delta = float(np.abs(actual - expected[start:stop]).max(initial=0))
        max_error = max(max_error, delta)
        equal &= bool(np.array_equal(actual, expected[start:stop]))
    if not equal:
        raise ValueError(f"Export changed cached predictions: max_abs_error={max_error}")
    manifest = {
        "status": "complete", "dataset": cache["dataset"], "format": FORMAT,
        "adapter": str(adapter_path), "adapter_sha256": sha256(adapter_path),
        "adapter_size_bytes": adapter_path.stat().st_size,
        "directions_shape": list(directions.shape), "directions_dtype": "float64",
        "strength": loaded.strength, "ridge_penalty": penalty,
        "class_names": list(loaded.class_names), "prompts": list(loaded.prompts),
        "provenance": loaded.metadata,
        "cached_prediction_equivalence": {
            "passed": True, "bitwise_equal": equal, "max_abs_error": max_error,
            "sample_count": len(expected), "chunk_size": 512, "test_labels_read": False},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RAW / "cache_loss_tiebreak")
    parser.add_argument("--output", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for dataset in args.datasets:
        reports[dataset] = export_dataset(args.run_root / dataset, args.output / dataset)
        print(f"{dataset}: {reports[dataset]['adapter_size_bytes']} bytes; all cached predictions bitwise equal", flush=True)
    write_json(args.output / "manifest.json", {
        "status": "complete", "method_family": "class_local weighted ridge least-squares",
        "datasets": reports})


if __name__ == "__main__":
    main()
