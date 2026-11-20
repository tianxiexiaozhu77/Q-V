#!/usr/bin/env python3
"""Fit one support draw and commit predictions before any test labels are read."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from data_interface import load_unit, object_sha256, sha256
from experiment import METHODS, load_snapshot, read_json, verify_fit_commit, write_json
from taskres_adapter import TaskResModel, fit_taskres
from sstext_adapter import SStextModel, fit_sstext


def fit_unit(run, model, dataset, seed, device="cpu"):
    # A worker may survive a controller crash. Keep a per-unit lock so resume
    # waits for that worker and reuses its commit instead of overwriting it.
    run = Path(run).resolve()
    config = load_snapshot(run)["config"]
    if model not in config["models"] or dataset not in config["datasets"] or seed not in config["seeds"]:
        raise ValueError("Unit outside the frozen experiment scope")
    folder = run / f"{model}_{dataset}/seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _fit_unit(run, model, dataset, seed, device)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _fit_unit(run, model, dataset, seed, device="cpu"):
    run = Path(run).resolve()
    snapshot = load_snapshot(run)
    config = snapshot["config"]
    if model not in config["models"] or dataset not in config["datasets"] or seed not in config["seeds"]:
        raise ValueError("Unit outside the frozen experiment scope")
    unit = load_unit(model, dataset, seed, config["catalog_path"], config["source_run"])
    expected = snapshot["unit_manifests"][unit.unit_id]
    if unit.manifest() != expected:
        raise ValueError("Dataset/cache provenance differs from preflight")
    folder = run / unit.unit_id
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_sha = sha256(run / "experiment_snapshot.json")
    if (folder / "fit_commit.json").exists():
        verify_fit_commit(folder, snapshot, snapshot_sha, unit.manifest())
        return
    if (run / "predictions_complete.json").exists():
        raise ValueError("Do not alter a run after predictions have been sealed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Explicit copies avoid mutating the authenticated read-only source arrays.
    common = dict(support_features=unit.support_features.copy(), labels=unit.labels.copy(),
                  queries_positive=unit.positive_text_raw.copy(),
                  queries_negative=unit.negative_text_raw.copy(), device=device)
    scale = snapshot["backbone_scales"][model]
    started = time.perf_counter()
    print(f"Fitting {unit.unit_id} on {device}: TaskRes grouped support CV", flush=True)
    taskres, task_info = fit_taskres(**common, groups=unit.groups, folds=unit.folds,
        seed=seed, logit_scale=scale["multiplier"], **config["taskres"])
    print(f"Fitting {unit.unit_id}: SS-Text+ fixed official solver", flush=True)
    sstext, ss_info = fit_sstext(**common, logit_scale=scale["natural_log"], **config["sstext_plus"])
    # Freeze fit/selection info before any test prediction is evaluated.
    info = dict(unit_id=unit.unit_id, test_labels_opened=False,
                input_manifest_sha256=object_sha256(unit.manifest()),
                taskres=task_info, sstext_plus=ss_info,
                backbone_scale=scale, feature_definition=unit.provenance["feature_definition"],
                text_base_policy=config["text_base_policy"],
                single_view_methods=["original", "query", "feature_text_zero_shot", "taskres", "sstext_plus"],
                two_view_methods=["view", "query_view"],
                fit_seconds=time.perf_counter()-started)
    write_json(folder / "fit_info.json", info)
    # Both exported states are array-only and reload without pickle.
    np.savez_compressed(folder / "taskres_state.npz",
                        **{k: np.asarray(v) for k, v in taskres.export().items()})
    np.savez_compressed(folder / "sstext_state.npz", **sstext.export())
    test_features = unit.test_features.copy()
    predictions = dict(unit.native_predictions)
    predictions["taskres"] = taskres.predict(test_features)
    predictions["sstext_plus"] = sstext.predict(test_features)
    zero = TaskResModel(unit.positive_text_raw, unit.negative_text_raw,
                       logit_scale=scale["multiplier"], residual_scale=config["taskres"]["residual_scale"]).to(device)
    predictions["feature_text_zero_shot"] = zero.predict(test_features)
    # A saved classifier must reproduce its predictions; do not silently save
    # incompatible serialization or change the prediction score domain.
    with np.load(folder / "taskres_state.npz", allow_pickle=False) as state:
        restored = TaskResModel.from_export({k: state[k] for k in state.files}, device=device)
        np.testing.assert_array_equal(predictions["taskres"], restored.predict(test_features))
    with np.load(folder / "sstext_state.npz", allow_pickle=False) as state:
        restored = SStextModel.from_export(state, device=device)
        np.testing.assert_array_equal(predictions["sstext_plus"], restored.predict(test_features))
    if set(predictions) != set(METHODS):
        raise ValueError("Unexpected method set")
    for name, values in predictions.items():
        if values.shape != (len(unit.test_paths), len(unit.classes)) or not np.isfinite(values).all():
            raise ValueError("Invalid predictions: " + name)
    np.savez_compressed(folder / "predictions.npz", image_paths=np.asarray(unit.test_paths),
        class_names=np.asarray(unit.classes), group_ids=np.asarray(unit.test_groups), **predictions)
    names = ("fit_info.json", "predictions.npz", "taskres_state.npz", "sstext_state.npz")
    write_json(folder / "fit_commit.json", dict(unit_id=unit.unit_id,
        unit_manifest_sha256=object_sha256(unit.manifest()), run_snapshot_sha256=snapshot_sha,
        source_sha256=snapshot["source_sha256"], artifacts={n: sha256(folder/n) for n in names},
        test_labels_opened=False, total_seconds=time.perf_counter()-started))
    print("Committed " + unit.unit_id, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    fit_unit(a.run, a.model, a.dataset, a.seed, a.device)


if __name__ == "__main__":
    main()
