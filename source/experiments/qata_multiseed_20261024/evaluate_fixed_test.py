#!/usr/bin/env python3
"""Evaluate five QaTa support draws after all validation thresholds are frozen."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.qata_multiseed_20261024.common import (
    BRANCHES, SEEDS, load_maps, load_rows, read_protocol, split_descriptor,
)
from experiments.qata_multiseed_20261024.select_thresholds import (
    choose_threshold, code_snapshot, publish_json, publish_npz,
)
from experiments.qata_validation_threshold_20261024 import dice_metrics as metrics

DISPLAY = dict(zip(BRANCHES, ("Original", "Positive query", "Q", "V", "Q+V")))
COMPARISONS = (
    ("Q_minus_Original", "query_refined", "original", "primary"),
    ("Q_minus_Positive", "query_refined", "positive_query", "primary"),
    ("Positive_minus_Original", "positive_query", "original", "secondary"),
    ("V_minus_Original", "tta_center_original", "original", "secondary"),
    ("QV_minus_Original", "tta_center_refined", "original", "secondary"),
    ("QV_minus_Q", "tta_center_refined", "query_refined", "secondary"),
    ("QV_minus_V", "tta_center_refined", "tta_center_original", "secondary"),
)


def verify_statistics_protocol(config: dict, replicates: int, bootstrap_seed: int) -> None:
    primary = [[left, right] for _, left, right, role in COMPARISONS if role == "primary"]
    secondary = [[left, right] for _, left, right, role in COMPARISONS if role == "secondary"]
    if config.get("primary_contrasts") != primary or config.get("secondary_contrasts") != secondary:
        raise ValueError("Paired contrasts differ from the frozen statistical protocol")
    bootstrap = config.get("bootstrap", {})
    if (bootstrap.get("replicates") != replicates or bootstrap.get("seed") != bootstrap_seed
            or bootstrap.get("unit") != "patient"
            or bootstrap.get("support_standard_deviation_ddof") != 1):
        raise ValueError("Bootstrap options differ from the frozen statistical protocol")


def verify_proof(proof: dict) -> None:
    """Verify every nested path/hash descriptor emitted by the map loader."""
    if "path" in proof and "sha256" in proof:
        if metrics.sha256(proof["path"]) != proof["sha256"]:
            raise ValueError(f"Frozen input changed: {proof['path']}")
    for value in proof.values():
        if isinstance(value, dict):
            verify_proof(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    verify_proof(item)


def verify_frozen_thresholds(path: Path, protocol_path: Path, config: dict) -> dict:
    """Gate all test annotation access on a complete verified 25-threshold file."""
    record = json.loads(path.read_text())
    if (record.get("status") != "frozen_before_fixed_test_scoring"
            or record.get("test_labels_used_for_selection") is not False
            or record.get("test_predictions_used_for_selection") is not False):
        raise ValueError("Missing validation-only frozen threshold status")
    if (record.get("seeds") != list(SEEDS) or record.get("branch_names") != list(BRANCHES)
            or not np.array_equal(record.get("threshold_grid"), metrics.THRESHOLDS)):
        raise ValueError("Frozen seeds, branches or threshold grid differ from protocol")
    if (metrics.sha256(protocol_path) != record["protocol"]["sha256"]
            or Path(record["protocol"]["path"]).resolve() != protocol_path.resolve()):
        raise ValueError("Protocol changed after threshold freezing")
    if record.get("code_sha256") != code_snapshot():
        raise ValueError("Selection/scoring implementation changed after threshold freezing")
    validation_path, _ = split_descriptor(protocol_path, config, "validation")
    if Path(record["validation_rows"]["path"]).resolve() != validation_path.resolve():
        raise ValueError("Frozen validation manifest differs from protocol")
    for field in ("validation_rows", "validation_predictions", "validation_curves", "validation_masks"):
        verify_proof(record[field])
    if set(record["thresholds"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("All five support seeds must be frozen before test scoring")
    if set(record["validation_predictions"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("Missing validation prediction provenance")
    rows = load_rows(protocol_path, config, "validation")
    expected_keys = {f"seed_{seed}__{branch}" for seed in SEEDS for branch in BRANCHES}
    with np.load(record["validation_curves"]["path"], allow_pickle=False) as data:
        if (set(data.files) != expected_keys | {"thresholds", "image_paths", "patient_ids", "seeds", "branch_names"}
                or not np.array_equal(data["thresholds"], metrics.THRESHOLDS)
                or data["image_paths"].astype(str).tolist() != [metrics.row_image(row) for row in rows]
                or data["patient_ids"].astype(str).tolist() != [metrics.row_patient(row) for row in rows]
                or data["seeds"].tolist() != list(SEEDS)
                or data["branch_names"].astype(str).tolist() != list(BRANCHES)):
            raise ValueError("Validation curves do not match the frozen split, seeds or branches")
        for seed in SEEDS:
            if set(record["thresholds"][str(seed)]) != set(BRANCHES):
                raise ValueError("Each seed requires all five native branches")
            for branch in BRANCHES:
                curves = data[f"seed_{seed}__{branch}"]
                if len(curves) != len(rows):
                    raise ValueError("Validation curve image count mismatch")
                expected = choose_threshold(curves)
                selected = record["thresholds"][str(seed)][branch]
                if any(selected.get(key) != value for key, value in expected.items()):
                    raise ValueError("Frozen threshold is not the predeclared validation optimum")
    return record


def summarize_scores(scores: np.ndarray, patient_ids: list[str], replicates: int = 2000,
                     bootstrap_seed: int = 20261024) -> tuple[dict, dict, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Keep support-draw SD separate from conditional patient bootstrap intervals."""
    x = np.asarray(scores)
    if (x.shape != (len(patient_ids), len(SEEDS), len(BRANCHES)) or not len(x)
            or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)):
        raise ValueError("Expected [images, five support seeds, five branches] Dice in [0,1]")
    if replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    # Each point uses the author's FP32 image mean, then the five values are
    # averaged in FP64. This is not a five-query prediction ensemble.
    per_seed = np.stack([metrics.author_mean(x[:, index, :]) for index in range(len(SEEDS))]).astype(np.float64)
    values, draws, patients = metrics.patient_bootstrap(
        x.reshape(len(x), -1), patient_ids, replicates, bootstrap_seed)
    boot = values.reshape(replicates, len(SEEDS), len(BRANCHES))
    boot_seed_mean = boot.mean(axis=1)
    means = per_seed.mean(axis=0)
    sds = per_seed.std(axis=0, ddof=1)
    intervals = np.quantile(boot_seed_mean, [.025, .975], axis=0)
    gain_intervals = np.quantile(boot_seed_mean - boot_seed_mean[:, :1], [.025, .975], axis=0)
    branches = {}
    for index, branch in enumerate(BRANCHES):
        branches[branch] = {
            "mean_test_dice": float(means[index]), "support_seed_sd": float(sds[index]),
            "mean_test_dice_ci95": intervals[:, index].tolist(),
            "mean_gain_over_original": float(means[index] - means[0]),
            "mean_gain_over_original_ci95": gain_intervals[:, index].tolist(),
            "per_seed_test_dice": per_seed[:, index].tolist(),
            "seeds_improved_over_original": int((per_seed[:, index] > per_seed[:, 0]).sum()),
        }
    comparisons = {}
    for name, left, right, role in COMPARISONS:
        a, b = BRANCHES.index(left), BRANCHES.index(right)
        deltas = per_seed[:, a] - per_seed[:, b]
        comparisons[name] = {
            "method": left, "reference": right, "role": role,
            "mean_delta_dice": float(deltas.mean()), "support_seed_delta_sd": float(deltas.std(ddof=1)),
            "delta_ci95": np.quantile(boot_seed_mean[:, a] - boot_seed_mean[:, b], [.025, .975]).tolist(),
            "per_seed_delta_dice": deltas.tolist(), "positive_seed_count": int((deltas > 0).sum()),
        }
    return branches, comparisons, boot, draws, patients, per_seed


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(protocol: Path, maps_root: Path, threshold_file: Path, output: Path,
             replicates: int = 2000, bootstrap_seed: int = 20261024,
             batch_size: int = 32) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty fixed-test output directory; results are not overwritten")
    if batch_size < 1 or replicates < 1:
        raise ValueError("Batch size and bootstrap replicates must be positive")
    protocol_path, config = read_protocol(protocol)
    threshold_file = threshold_file.resolve()
    # This gate runs before any test annotations, native maps or masks are opened.
    frozen = verify_frozen_thresholds(threshold_file, protocol_path, config)
    verify_statistics_protocol(config, replicates, bootstrap_seed)
    threshold_sha = metrics.sha256(threshold_file)
    validation_rows = load_rows(protocol_path, config, "validation")
    rows = load_rows(protocol_path, config, "test")
    test_path, _ = split_descriptor(protocol_path, config, "test")
    patient_ids = [metrics.row_patient(row) for row in rows]
    if set(patient_ids) & {metrics.row_patient(row) for row in validation_rows}:
        raise ValueError("Validation/test patient overlap")
    maps_by_seed, proofs = {}, {}
    for seed in SEEDS:
        maps, proof = load_maps(maps_root, protocol_path, config, "test", seed)
        validation_proof = frozen["validation_predictions"][str(seed)]
        for key in ("manifest", "query_fit"):
            if proof.get(key) != validation_proof.get(key) or key not in proof:
                raise ValueError(f"Validation/test maps do not share frozen {key}: seed={seed}")
        maps_by_seed[seed], proofs[str(seed)] = maps, proof
    # All 25 thresholds and prediction identities are verified before test masks.
    masks, mask_proof = metrics.load_positive_masks(rows)
    scores = np.empty((len(rows), len(SEEDS), len(BRANCHES)), dtype=np.float32)
    references = np.empty_like(scores)
    per_seed_rows = []
    for si, seed in enumerate(SEEDS):
        for bi, branch in enumerate(BRANCHES):
            selected = frozen["thresholds"][str(seed)][branch]
            threshold = selected["threshold"]
            prescribed = np.unique(np.asarray([.5, threshold]))
            values = metrics.score_maps(maps_by_seed[seed][branch], masks, prescribed, batch_size)
            scores[:, si, bi] = values[:, int(np.flatnonzero(prescribed == threshold)[0])]
            references[:, si, bi] = values[:, int(np.flatnonzero(prescribed == .5)[0])]
        del maps_by_seed[seed]
    if metrics.sha256(threshold_file) != threshold_sha or code_snapshot() != frozen["code_sha256"]:
        raise ValueError("Frozen thresholds or scoring code changed during evaluation")
    branches, comparisons, boot, draws, patients, per_seed = summarize_scores(
        scores, patient_ids, replicates, bootstrap_seed)
    reference_means = np.stack([metrics.author_mean(references[:, i, :]) for i in range(len(SEEDS))]).astype(np.float64)
    for si, seed in enumerate(SEEDS):
        for bi, branch in enumerate(BRANCHES):
            entry = frozen["thresholds"][str(seed)][branch]
            per_seed_rows.append({"seed": seed, "method": branch, "threshold": entry["threshold"],
                                  "validation_dice_percent": 100 * entry["validation_dice"],
                                  "test_dice_percent": 100 * per_seed[si, bi],
                                  "delta_over_original_pp": 100 * (per_seed[si, bi] - per_seed[si, 0]),
                                  "fixed_0_5_dice_percent": 100 * reference_means[si, bi]})
    output.mkdir(parents=True, exist_ok=True)
    publish_npz(output / "per_image_dice.npz", scores=scores, fixed_0_5_scores=references,
                support_seeds=np.asarray(SEEDS), branch_names=np.asarray(BRANCHES),
                patient_ids=np.asarray(patient_ids), image_paths=np.asarray([metrics.row_image(row) for row in rows]))
    publish_npz(output / "patient_bootstrap.npz", dice=boot, draws=draws,
                support_seeds=np.asarray(SEEDS), branch_names=np.asarray(BRANCHES), patients=np.asarray(patients))
    report = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "AlphaRAD", "dataset": "QaTa-COV19-v2", "support_seeds": list(SEEDS),
        "class_names": ["covid"], "test_images": len(rows), "test_patients": len(patients),
        "validation_images": len(validation_rows),
        "validation_patients": len({metrics.row_patient(row) for row in validation_rows}),
        "eta": .3, "crop_fraction": .8, "branches": branches, "paired_comparisons": comparisons,
        "per_seed_results": per_seed_rows,
        "protocol": {"path": str(protocol_path), "sha256": metrics.sha256(protocol_path)},
        "thresholds": {"path": str(threshold_file), "sha256": threshold_sha},
        "test_rows": {"path": str(test_path), "sha256": metrics.sha256(test_path)},
        "test_predictions": proofs, "test_masks": mask_proof,
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed,
                      "unit": "explicit BIMCV subject ID; retain all images of sampled patients",
                      "aggregation": "image-equal Dice for each support seed, then arithmetic mean over five seeds",
                      "paired_across_all_support_seeds_and_branches": True,
                      "conditional_on": "the five fixed support sets, fixed validation split and all selected thresholds",
                      "includes_future_support_population_uncertainty": False,
                      "includes_validation_threshold_selection_uncertainty": False,
                      "multiple_comparison_correction": False, "percentiles": [2.5, 97.5]},
        "annotation_budget": frozen["annotation_budget"], "test_threshold_search_performed": False,
        "fixed_0_5_reference": {"threshold": .5, "primary_result": False,
                                "mean_test_dice": dict(zip(BRANCHES, reference_means.mean(axis=0).tolist())),
                                "rule": "Predeclared secondary reference, never substituted for validation-selected results using test scores."},
        "code_sha256": code_snapshot(),
        "limitations": [
            "Validation pixel masks are additional supervision beyond the 10 support image labels per seed.",
            "Positive-only foreground Dice does not measure false-positive regions on normal test images.",
            "Normal support controls lack reliable patient IDs; exact-image separation does not establish full patient isolation.",
            "Patient confidence intervals condition on these five supports and validation thresholds; seed SD separately describes these five support draws.",
            "This test set previously participated in method development; resampling does not remove that selection bias.",
        ],
    }
    write_csv(output / "PER_SEED_RESULTS.csv", list(per_seed_rows[0]), per_seed_rows)
    mean_rows = []
    for branch, result in branches.items():
        mean_rows.append({"method": branch, "mean_test_dice_percent": 100 * result["mean_test_dice"],
                          "support_seed_sd_pp": 100 * result["support_seed_sd"],
                          "mean_delta_dice_pp": 100 * result["mean_gain_over_original"],
                          "delta_ci95_lower_pp": 100 * result["mean_gain_over_original_ci95"][0],
                          "delta_ci95_upper_pp": 100 * result["mean_gain_over_original_ci95"][1],
                          "positive_seed_count": result["seeds_improved_over_original"]})
    write_csv(output / "MEAN_SD_COMPARISON.csv", list(mean_rows[0]), mean_rows)
    paired_rows, seed_pair_rows = [], []
    for name, result in comparisons.items():
        paired_rows.append({"comparison": name, "method": result["method"], "reference": result["reference"],
                            "role": result["role"], "mean_delta_dice_pp": 100 * result["mean_delta_dice"],
                            "support_seed_delta_sd_pp": 100 * result["support_seed_delta_sd"],
                            "delta_ci95_lower_pp": 100 * result["delta_ci95"][0],
                            "delta_ci95_upper_pp": 100 * result["delta_ci95"][1],
                            "positive_seed_count": result["positive_seed_count"]})
        seed_pair_rows.extend({"seed": seed, "comparison": name, "role": result["role"],
                               "delta_dice_pp": 100 * value}
                              for seed, value in zip(SEEDS, result["per_seed_delta_dice"]))
    write_csv(output / "PAIRED_COMPARISONS.csv", list(paired_rows[0]), paired_rows)
    write_csv(output / "PER_SEED_PAIRED_COMPARISONS.csv", list(seed_pair_rows[0]), seed_pair_rows)
    text = ["AlphaRAD / QaTa 五支持种子验证阈值空间评估", "",
            f"支持种子：{list(SEEDS)}；验证 {len(validation_rows)} 张；测试 {len(rows)} 张 / {len(patients)} 患者。",
            "各种子、各分支仅用验证集选阈值；全部25个阈值冻结后才读取测试掩码。",
            "Dice 是阳性图像、前景、图像等权指标；均值为五次适配指标的平均，不是预测集成。", "",
            "方法 | 五种子 Dice 均值 ± SD (%) | 对 Original 平均增益 [95% CI] (百分点)",
            "--- | ---: | ---:"]
    for branch, result in branches.items():
        lo, hi = result["mean_gain_over_original_ci95"]
        text.append(f"{DISPLAY[branch]} | {100*result['mean_test_dice']:.2f} ± {100*result['support_seed_sd']:.2f} | "
                    f"{100*result['mean_gain_over_original']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    text += ["", "预先指定的配对比较（前两项为主要比较；其余为辅助分析）：",
             "比较 | 平均差值 [95% CI] (百分点) | 正向种子数", "--- | ---: | ---:"]
    for name, result in comparisons.items():
        lo, hi = result["delta_ci95"]
        text.append(f"{name} | {100*result['mean_delta_dice']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | "
                    f"{result['positive_seed_count']}/5")
    text += ["", f"区间：{replicates} 次患者级配对 bootstrap，所有种子和分支共享抽样。",
             "区间条件于当前五组支持、固定验证集与阈值；不包含未来支持集或验证阈值选择的不确定性。",
             "种子间波动使用样本标准差（ddof=1），逐种子阈值和 Dice 保存在 PER_SEED_RESULTS.csv。",
             "正常支持图缺少可靠患者编号，只核实精确图像分离；不能声称支持与验证/测试完全患者隔离。",
             f"额外空间监督：{len(validation_rows)} 张验证图掩码；该测试集曾参与方法开发。",
             "固定0.5阈值参照单独保存在 JSON 和逐种子 CSV，不按测试成绩替换主结果。"]
    (output / "RESULTS.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    # Completion marker is published only after all machine-readable outputs.
    publish_json(output / "RESULTS.json", report)
    print("\n".join(text), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--maps", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate(args.protocol, args.maps, args.thresholds, args.output,
             args.replicates, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
