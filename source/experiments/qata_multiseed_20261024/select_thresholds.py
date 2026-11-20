#!/usr/bin/env python3
"""Freeze all five-seed QaTa thresholds using the fixed validation split only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qata_multiseed_20261024.common import (
    BRANCHES, SEEDS, load_maps, load_rows, read_protocol, split_descriptor,
)
from experiments.qata_validation_threshold_20261024 import dice_metrics as metrics


def publish_json(path: Path, record: dict) -> None:
    """Atomically publish a complete immutable record without replacing a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is atomic and fails if the target exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_snapshot() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = [directory / name for name in
             ("common.py", "select_thresholds.py", "evaluate_fixed_test.py")]
    paths.append(Path(metrics.__file__).resolve())
    return {str(path): metrics.sha256(path) for path in paths}


def choose_threshold(curves: np.ndarray) -> dict:
    values = np.asarray(curves)
    if (values.ndim != 2 or values.shape[1] != len(metrics.THRESHOLDS)
            or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("Expected finite [validation images, 101 thresholds] Dice in [0,1]")
    means = metrics.author_mean(values)
    best = int(np.argmax(means))  # Ascending grid, exact ties -> smallest tau.
    return {"threshold": float(metrics.THRESHOLDS[best]), "threshold_index": best,
            "validation_dice": float(means[best])}


def select_thresholds(protocol: Path, maps_root: Path, output: Path,
                      batch_size: int = 32) -> dict:
    if output.exists():
        raise FileExistsError("Frozen thresholds cannot be overwritten; use a new run directory")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    protocol_path, config = read_protocol(protocol)
    start_code = code_snapshot()
    rows = load_rows(protocol_path, config, "validation")
    validation_path, _ = split_descriptor(protocol_path, config, "validation")
    patient_ids = [metrics.row_patient(row) for row in rows]
    # No test annotations, masks, scores or map arrays are opened in this stage.
    masks, mask_proof = metrics.load_positive_masks(rows)
    all_curves, selected, proofs = {}, {}, {}
    for seed in SEEDS:
        maps, proof = load_maps(maps_root, protocol_path, config, "validation", seed)
        proofs[str(seed)] = proof
        selected[str(seed)] = {}
        for branch in BRANCHES:
            curve = metrics.score_maps(maps[branch], masks, metrics.THRESHOLDS, batch_size)
            all_curves[f"seed_{seed}__{branch}"] = curve
            entry = choose_threshold(curve)
            entry.update(validation_images=len(rows), validation_patients=len(set(patient_ids)))
            selected[str(seed)][branch] = entry
            print(f"seed={seed} {branch}: validation tau={entry['threshold']:.2f} "
                  f"Dice={entry['validation_dice']:.8f}", flush=True)
    if code_snapshot() != start_code:
        raise ValueError("Selection/scoring code changed during validation selection")
    curves_path = output.with_name(output.stem + "_validation_curves.npz")
    publish_npz(curves_path, thresholds=metrics.THRESHOLDS, **all_curves,
                image_paths=np.asarray([metrics.row_image(row) for row in rows]),
                patient_ids=np.asarray(patient_ids), seeds=np.asarray(SEEDS),
                branch_names=np.asarray(BRANCHES))
    record = {
        "schema_version": 1, "status": "frozen_before_fixed_test_scoring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS), "branch_names": list(BRANCHES), "thresholds": selected,
        "threshold_grid": metrics.THRESHOLDS.tolist(),
        "threshold_rule": "maximum image-mean foreground validation Dice; smallest threshold on exact ties",
        "threshold_comparison": "float32 probability > float32 threshold",
        "test_labels_used_for_selection": False, "test_predictions_used_for_selection": False,
        "protocol": {"path": str(protocol_path), "sha256": metrics.sha256(protocol_path)},
        "validation_rows": {"path": str(validation_path), "sha256": metrics.sha256(validation_path)},
        "validation_predictions": proofs, "validation_masks": mask_proof,
        "validation_curves": {"path": str(curves_path.resolve()), "sha256": metrics.sha256(curves_path)},
        "code_sha256": start_code,
        "annotation_budget": {
            "support_image_labels_per_seed": 10,
            "positive_support_images_per_seed": 5, "negative_support_images_per_seed": 5,
            "additional_validation_images_with_pixel_masks": len(rows),
            "additional_validation_patients": len(set(patient_ids)),
            "note": "The same validation masks calibrate every branch/seed and are additional spatial supervision. Positive-only query uses only the five shared positive supports.",
        },
    }
    publish_json(output, record)
    print(f"Frozen all {len(SEEDS) * len(BRANCHES)} thresholds: {output}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path, help="Run maps root with a complete manifest")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    select_thresholds(args.protocol, args.maps, args.output, args.batch_size)


if __name__ == "__main__":
    main()
