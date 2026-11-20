#!/usr/bin/env python3
"""Select and freeze four QaTa thresholds using TRAIN-held-out validation only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from dice_metrics import (BRANCHES, THRESHOLDS, atomic_json, author_mean, load_maps,
                         load_positive_masks, load_rows, read_protocol, row_patient,
                         score_maps, sha256, split_descriptor)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--maps", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise FileExistsError("Frozen thresholds cannot be overwritten; use a fresh run directory")
    protocol_path, protocol = read_protocol(args.protocol)
    row_path, row_sha = split_descriptor(protocol_path, protocol, "validation")
    rows = load_rows(row_path, row_sha)
    # Deliberately never load test_rows or masks here.
    maps, prediction_proof = load_maps(args.maps, rows)
    masks, mask_audit = load_positive_masks(rows)
    curves, selected = {}, {}
    for branch in BRANCHES:
        image_dice = score_maps(maps[branch], masks, THRESHOLDS, args.batch_size)
        means = author_mean(image_dice)
        best = int(np.argmax(means))  # Ascending grid -> smallest threshold on ties.
        selected[branch] = dict(threshold=float(THRESHOLDS[best]), threshold_index=best,
                                validation_dice=float(means[best]),
                                validation_images=len(rows),
                                validation_patients=len({row_patient(r) for r in rows}))
        curves[branch] = image_dice
        print(f"{branch}: validation threshold={THRESHOLDS[best]:.2f}, Dice={means[best]:.8f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    curves_path = args.output.with_name(args.output.stem + "_validation_curves.npz")
    if curves_path.exists():
        raise FileExistsError(curves_path)
    np.savez_compressed(curves_path, thresholds=THRESHOLDS, **curves,
                        patient_ids=np.asarray([row_patient(r) for r in rows], dtype=str))
    record = {
        "schema_version": 1, "status": "frozen_before_fixed_test_scoring",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_data": "validation split held out by patient from QaTa training data",
        "test_labels_used_for_selection": False,
        "test_predictions_used_for_selection": False,
        "threshold_rule": "maximum image-mean foreground Dice on validation; smallest threshold on ties",
        "threshold_grid": THRESHOLDS.tolist(), "threshold_comparison": "float32 probability > float32 threshold",
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "validation_rows": {"path": str(row_path), "sha256": sha256(row_path)},
        "validation_predictions": prediction_proof,
        "validation_masks": mask_audit,
        "validation_curves": {"path": str(curves_path.resolve()), "sha256": sha256(curves_path)},
        "branches": selected,
        "code_sha256": {name: sha256(Path(__file__).parent / name)
                        for name in ("dice_metrics.py", "select_thresholds.py")},
        "annotation_budget": {"additional_validation_images_with_pixel_masks": len(rows),
                              "additional_validation_patients": len({row_patient(r) for r in rows}),
                              "note": "These validation segmentation masks are additional supervision beyond the query support image labels."},
    }
    atomic_json(args.output, record, exclusive=True)
    print(f"Frozen thresholds: {args.output} sha256={sha256(args.output)}", flush=True)


if __name__ == "__main__":
    main()
