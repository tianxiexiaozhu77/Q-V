#!/usr/bin/env python3
"""Score the unchanged QaTa test set at previously frozen validation thresholds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from dice_metrics import (BRANCHES, DISPLAY, atomic_json, author_mean, load_maps,
                         load_positive_masks, load_rows, patient_bootstrap, read_protocol,
                         row_image, row_patient, score_maps, sha256, split_descriptor)


def verify_frozen_thresholds(path: Path, protocol_path: Path) -> dict:
    record = json.loads(path.read_text())
    if record["status"] != "frozen_before_fixed_test_scoring":
        raise ValueError("Missing frozen validation threshold status")
    if record["test_labels_used_for_selection"] or record["test_predictions_used_for_selection"]:
        raise ValueError("Thresholds must be selected without test labels or predictions")
    if sha256(protocol_path) != record["protocol"]["sha256"]:
        raise ValueError("Protocol changed after threshold freezing")
    for field in ("validation_rows", "validation_predictions", "validation_curves"):
        ref = record[field]
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError(f"Validation artifact changed after freezing: {field}")
    for name in ("dice_metrics.py", "select_thresholds.py"):
        if sha256(Path(__file__).parent / name) != record["code_sha256"][name]:
            raise ValueError(f"Scoring/selection implementation changed after freezing: {name}")
    if set(record["branches"]) != set(BRANCHES):
        raise ValueError("Missing or extra frozen model branch")
    for branch in BRANCHES:
        selected = record["branches"][branch]
        if selected["threshold"] != record["threshold_grid"][selected["threshold_index"]]:
            raise ValueError("Threshold and frozen validation grid do not agree")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path,
                        help="test_cache_manifest.json, or test native_maps.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh fixed-test output directory; do not overwrite results")
    protocol_path, protocol = read_protocol(args.protocol)
    # Required temporal gate: validate persisted validation thresholds BEFORE
    # reading any test row annotations, native map arrays or source masks.
    frozen = verify_frozen_thresholds(args.thresholds.resolve(), protocol_path)
    threshold_sha = sha256(args.thresholds)
    test_path, test_sha = split_descriptor(protocol_path, protocol, "test")
    rows = load_rows(test_path, test_sha)
    val_rows = load_rows(Path(frozen["validation_rows"]["path"]), frozen["validation_rows"]["sha256"])
    patient_ids = [row_patient(r) for r in rows]
    if set(patient_ids) & {row_patient(r) for r in val_rows}:
        raise ValueError("Validation/test patient overlap")
    maps, prediction_proof = load_maps(args.maps, rows)
    masks, mask_audit = load_positive_masks(rows)
    scores = np.empty((len(rows), len(BRANCHES)), dtype=np.float32)
    reference_scores = np.empty_like(scores)
    for column, branch in enumerate(BRANCHES):
        threshold = float(frozen["branches"][branch]["threshold"])
        # The 0.5 reference is prescribed by the pre-existing research plan,
        # not selected from test performance and never substituted for the main result.
        prescribed = np.unique(np.asarray([.5, threshold]))
        values = score_maps(maps[branch], masks, prescribed, args.batch_size)
        scores[:, column] = values[:, int(np.flatnonzero(prescribed == threshold)[0])]
        reference_scores[:, column] = values[:, int(np.flatnonzero(prescribed == .5)[0])]
        print(f"{branch}: fixed threshold={threshold:.2f}, test Dice={author_mean(scores[:,column:column+1])[0]:.8f}", flush=True)
    if sha256(args.thresholds) != threshold_sha:
        raise ValueError("Frozen thresholds changed during test evaluation")
    means = author_mean(scores)
    reference_means = author_mean(reference_scores)
    boot, draws, patients = patient_bootstrap(scores, patient_ids, args.replicates, args.seed)
    intervals = np.quantile(boot, [.025, .975], axis=0)
    gains = boot - boot[:, :1]
    gain_intervals = np.quantile(gains, [.025, .975], axis=0)
    branch_results = {}
    for column, branch in enumerate(BRANCHES):
        branch_results[branch] = dict(
            threshold=frozen["branches"][branch]["threshold"],
            validation_dice=frozen["branches"][branch]["validation_dice"],
            test_dice=float(means[column]), test_dice_ci95=intervals[:, column].tolist(),
            gain_over_original=float(means[column] - means[0]),
            gain_over_original_ci95=gain_intervals[:, column].tolist(),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "per_image_dice.npz", scores=scores, fixed_0_5_scores=reference_scores,
                        branch_names=np.asarray(BRANCHES),
                        image_paths=np.asarray([row_image(r) for r in rows]),
                        patient_ids=np.asarray(patient_ids))
    np.savez_compressed(args.output / "patient_bootstrap.npz", dice=boot, delta_dice=gains,
                        draws=draws, patients=np.asarray(patients), branch_names=np.asarray(BRANCHES))
    report = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "QaTa-COV19-v2", "model": "AlphaRAD", "class_names": ["covid"],
        "test_images": len(rows), "test_patients": len(patients),
        "validation_images": len(val_rows), "validation_patients": len({row_patient(r) for r in val_rows}),
        "eta": 0.3, "crop_fraction": 0.8,
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "thresholds": {"path": str(args.thresholds.resolve()), "sha256": threshold_sha},
        "test_rows": {"path": str(test_path), "sha256": sha256(test_path)},
        "test_predictions": prediction_proof, "test_masks": mask_audit,
        "branches": branch_results,
        "fixed_0_5_reference": {
            "threshold": .5, "validation_masks_used_to_set_threshold": False,
            "predeclared_source": "baselines/paper_draft_zh/RESEARCH_VALIDATION_PLAN.txt, spatial validation plan",
            "rule": "Report all four branches separately; never choose between this reference and validation-calibrated primary results using test Dice.",
            "branches": {branch: {"test_dice": float(reference_means[i]),
                                  "gain_over_original": float(reference_means[i] - reference_means[0])}
                         for i, branch in enumerate(BRANCHES)}},
        "bootstrap": {"replicates": args.replicates, "seed": args.seed,
                      "unit": "explicit BIMCV subject ID", "aggregation": "equal image weights, all images from each sampled patient retained",
                      "paired_across_branches": True, "percentiles": [2.5, 97.5],
                      "conditional_on": "fixed support set and fixed validation-selected thresholds",
                      "includes_threshold_selection_uncertainty": False,
                      "valid_replicates": int(np.isfinite(boot).all(axis=1).sum())},
        "annotation_budget": frozen["annotation_budget"],
        "test_threshold_search_performed": False,
        "code_sha256": {name: sha256(Path(__file__).parent / name)
                        for name in ("dice_metrics.py", "select_thresholds.py", "evaluate_fixed_test.py")},
        "limitations": ["Validation thresholds use extra pixel-level segmentation annotations beyond support image labels.",
                        "This is positive-only foreground Dice; it does not measure false-positive masks on normal patients.",
                        "The frozen support/query and test views were previously explored; patient bootstrap does not correct method-selection bias.",
                        "CI is conditional on the one fixed validation split/threshold and support set; it does not include threshold/support resampling."],
    }
    atomic_json(args.output / "RESULTS.json", report)
    with (args.output / "RESULTS.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "threshold", "validation_dice_percent", "test_dice_percent", "delta_dice_pp", "delta_ci95_lower_pp", "delta_ci95_upper_pp"])
        for branch in BRANCHES:
            r = branch_results[branch]
            writer.writerow([DISPLAY[branch], r["threshold"], 100*r["validation_dice"], 100*r["test_dice"],
                             100*r["gain_over_original"], *[100*v for v in r["gain_over_original_ci95"]]])
    text = ["QaTa-COV19-v2：训练患者验证集选阈值，固定测试集分割评估", "",
            f"AlphaRAD，η=0.3，中心80%双视图；验证 {len(val_rows)} 张 / {report['validation_patients']} 患者；测试 {len(rows)} 张 / {len(patients)} 患者。",
            "验证集在每分支 0:0.01:1 网格独立选阈值，持久化冻结后才打开测试标注；并列取最小阈值。",
            "主指标为作者阳性样本、前景、图像等权 Dice。数值单位%，增益单位百分点。", "",
            "方法 | 验证选定阈值 | 验证 Dice | 固定测试 Dice | 相对 Original 增益 [95% CI]",
            "--- | ---: | ---: | ---: | ---:"]
    for branch in BRANCHES:
        r = branch_results[branch]
        lo, hi = r["gain_over_original_ci95"]
        text.append(f"{DISPLAY[branch]} | {r['threshold']:.2f} | {100*r['validation_dice']:.2f} | {100*r['test_dice']:.2f} | {100*r['gain_over_original']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    text += ["", "预先约定的固定阈值0.5参照（阈值不使用新增验证掩码；与主结果分开报告，不择优替换）：",
             "方法 | 固定阈值 | 测试 Dice | 相对 Original 增益", "--- | ---: | ---: | ---:"]
    for i, branch in enumerate(BRANCHES):
        text.append(f"{DISPLAY[branch]} | 0.50 | {100*reference_means[i]:.2f} | {100*(reference_means[i] - reference_means[0]):+.2f}")
    text += ["", f"置信区间：{args.replicates} 次患者级配对 bootstrap，seed={args.seed}，四分支共享抽样。",
             "区间条件于固定支持集、固定验证集及其选定阈值，不含支持或阈值选择波动，也不修正先前测试集方法选择偏差。",
             f"标注预算：另使用 {len(val_rows)} 张验证图的像素级掩码，不能将本评估协议描述为仅使用10张支持图的图像标签。",
             "测试集没有重选阈值；历史测试集最优Dice仅可作为不同协议结果，不能直接替代本表。",
             "此指标仅评估有病灶标注的阳性图像，不涵盖正常图像上的假阳性区域。", "",
             f"冻结阈值：{args.thresholds.resolve()}", f"阈值SHA256：{threshold_sha}"]
    (args.output / "RESULTS.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)


if __name__ == "__main__":
    main()
