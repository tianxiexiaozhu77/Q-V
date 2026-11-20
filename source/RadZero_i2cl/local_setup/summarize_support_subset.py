#!/usr/bin/env python3
"""Audit the three fixed-test one-shot runs and compare to five-shot weighted I2CL."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from audit_i2cl import (
    SUPPORT_WEIGHTED_LOSS, audit, audit_dataset, close, identity, read_json,
    read_predictions, require, sha256, verify_metrics, write_json,
)
from i2cl_data import load_spec


DATASETS = ("ChestXDet10", "ChestXray14", "OpenI")
METRICS = ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc")


def percentage(value):
    return "未定义" if value is None else f"{100 * value:.2f}%"


def delta(current, previous):
    return {key: None if current[key] is None or previous[key] is None else
            100 * (current[key] - previous[key]) for key in METRICS}


def summarize(run_dir, parent_run, parent_summary, output):
    run_dir = Path(run_dir).resolve()
    parent_run = Path(parent_run).resolve()
    parent_summary = Path(parent_summary).resolve()
    output = Path(output).resolve()
    require(run_dir != parent_run, "Current run must differ from the parent run")
    require(output not in (parent_run, parent_summary.parent),
            "The new summary must not overwrite the previous experiment")
    launcher = read_json(run_dir / "launcher.json")
    require(set(launcher["datasets"]) == set(DATASETS),
            "This summary requires exactly ChestXDet10, ChestXray14 and OpenI")
    require(launcher["status"] == "complete" and not launcher["is_smoke"],
            "All three full evaluations must be complete")
    report = audit(run_dir)
    parent_launcher = read_json(parent_run / "launcher.json")
    require(parent_launcher["status"] == "complete" and not parent_launcher["is_smoke"],
            "Parent is not a completed full experiment")
    previous_summary = read_json(parent_summary)
    require(previous_summary["status"] == "complete_and_audited",
            "Parent summary was not completed and audited")
    for path, expected in parent_launcher["source_sha256"].items():
        require(sha256(path) == expected, f"Parent launcher source changed: {path}")

    datasets, previous_audits = {}, {}
    for name in DATASETS:
        row = report["datasets"][name]
        parent_job = parent_launcher["jobs"][name]
        require(parent_job["status"] == "complete" and parent_job["exit_code"] == 0,
                f"{name}: parent job failed")
        # audit_dataset only reads the old experiment; no old report is rewritten.
        old = audit_dataset(parent_run / name, parent_job, allow_smoke=False)
        previous_audits[name] = old
        saved_old = previous_summary["datasets"][name]
        verify_metrics(saved_old["i2cl_metrics"], old["i2cl_metrics"],
                       f"{name}: previous saved I2CL summary")
        verify_metrics(saved_old["baseline_metrics"], old["baseline_metrics"],
                       f"{name}: previous saved baseline summary")
        require(row["protocol"] == "support_subset_fixed_test" and
                row["support_subset_verification"]["status"] == "passed",
                f"{name}: subset provenance was not verified")
        require(row["shots_per_class"] == 1 and old["shots_per_class"] == 5 and
                row["support_images"] * 5 == old["support_images"],
                f"{name}: wrong one-shot/five-shot support counts")
        require(row["loss"] == old["loss"] == SUPPORT_WEIGHTED_LOSS,
                f"{name}: runs use different loss definitions")
        require(row.get("regularization") == old.get("regularization"),
                f"{name}: runs use different coefficient regularization settings")
        for key, expected in (
            ("epochs", 100), ("grad_batch_size", 2), ("learning_rate", .01),
            ("weight_decay", .001), ("noise_scale", .001), ("trainable_scalars", 8),
        ):
            require(row[key] == old[key] == expected, f"{name}: changed training setting {key}")
        require(row["optimizer_steps"] * 5 == old["optimizer_steps"],
                f"{name}: unexpected optimizer-step ratio")
        require(row["base_tensor_digest"] == old["base_tensor_digest"],
                f"{name}: base model tensors differ")
        require(row["support_identity"] != old["support_identity"] and
                row["same_support_context_and_calibration"],
                f"{name}: support/context identity mismatch")
        current_config = read_json(run_dir / name / "run_config.json")
        previous_config = read_json(parent_run / name / "run_config.json")
        for key in ("source_sha256", "checkpoint", "model_config_sha256",
                    "preprocessor_sha256", "resolution", "precision", "sampling",
                    "checkpoint_selection", "class_names", "prompts"):
            require(current_config[key] == previous_config[key],
                    f"{name}: previous and current {key} differ")
        for key in ("seed", "eval_batch_size", "workers"):
            require(current_config["arguments"][key] == previous_config["arguments"][key],
                    f"{name}: changed execution setting {key}")
        current_split = Path(current_config["arguments"]["split_dir"])
        previous_split = Path(previous_config["arguments"]["split_dir"])
        parent_verification = row["support_subset_verification"]
        require(Path(parent_verification["parent_split_root"]).resolve() ==
                previous_split.resolve(), f"{name}: subset was not derived from this parent run")
        require(parent_verification["parent_support_sha256"] == old["support_sha256"],
                f"{name}: previous support checksum differs")
        test = load_spec(current_split, name, "test")
        old_test = load_spec(previous_split, name, "test")
        require(identity(test) == identity(old_test), f"{name}: test populations differ")
        current_baseline = read_predictions(run_dir / name / "baseline_predictions.csv", test)
        previous_baseline = read_predictions(parent_run / name / "baseline_predictions.csv", old_test)
        max_error = float(np.max(np.abs(current_baseline - previous_baseline)))
        require(np.allclose(current_baseline, previous_baseline, rtol=0, atol=1e-4),
                f"{name}: baseline logits changed beyond FP32 tolerance")
        for key in METRICS:
            close(row["baseline_metrics"][key], old["baseline_metrics"][key],
                  f"{name}: fixed-test baseline {key}")
        datasets[name] = {
            **row,
            "run_directory": str(run_dir / name),
            "previous_run_directory": str(parent_run / name),
            "previous_support_images": old["support_images"],
            "previous_optimizer_steps": old["optimizer_steps"],
            "previous_loss_weighting": old["loss_weighting"],
            "previous_i2cl_metrics": old["i2cl_metrics"],
            "delta_vs_previous_pp": delta(row["i2cl_metrics"], old["i2cl_metrics"]),
            "baseline_logits_bit_identical": bool(np.array_equal(
                current_baseline, previous_baseline)),
            "baseline_logits_max_absolute_difference": max_error,
        }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "parent_cpu_audit.json", {
        "status": "passed", "parent_run": str(parent_run),
        "scope": "Read-only re-audit of the three corresponding previous datasets",
        "datasets": previous_audits,
    })
    summary = {
        "status": "complete_and_audited",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "project": "RadZero_i2cl",
        "protocol": "support_subset_fixed_test",
        "run_directory": str(run_dir), "parent_run_directory": str(parent_run),
        "parent_summary": {"path": str(parent_summary), "sha256": sha256(parent_summary)},
        "audit": {"path": str(run_dir / "audit.json"),
                  "sha256": sha256(run_dir / "audit.json")},
        "summary_code_sha256": sha256(__file__),
        "metric_units": "AUROC/AUPRC fractions; deltas in percentage points",
        "auprc_method": "Trapezoidal area under precision-recall curve; not average precision",
        "comparison": {
            "test_csv_byte_identical_to_previous": True,
            "support": "One of the previous five anchor-assigned images per disease; seed 42",
            "unused_previous_support_returned_to_test": False,
            "epochs": 100, "gradient_batch_size": 2,
            "constant_learning_rate": .01, "weight_decay": .001, "noise_scale": .001,
            "trainable_scalars": 8, "loss": SUPPORT_WEIGHTED_LOSS,
            "regularization_by_dataset": {
                name: row.get("regularization") for name, row in datasets.items()
            },
            "pos_weight": "Recomputed per class from the new support negatives/positives",
            "context": "New unweighted CLS mean from exactly the new calibration support",
            "optimizer_steps": "One fifth of the previous run because epochs and batch size stay fixed",
            "checkpoint_selection": "final fixed epoch; no validation/test checkpoint selection",
        },
        "datasets": datasets,
        "limitations": [
            "Single seed 42; no confidence interval or repeated-support robustness claim.",
            "Fewer support images also reduce optimizer steps at fixed 100 epochs.",
            "Changes combine support content/size, rebuilt context, recalculated loss weights and fewer updates.",
            "Only classification AUROC/AUPRC evaluated; no localization or Dice claim.",
            "One anchor per class is an allocation quota, not exactly one positive label per class.",
            "Support is held out from released evaluation data; this is supervised few-shot adaptation.",
            "OpenI groups reports, not verified longitudinal patient identities; ChestXDet10 lacks complete patient IDs.",
        ],
    }
    write_json(output / "summary.json", summary)
    lines = [
        "RadZero_i2cl：每类 1 张支持图与每类 5 张支持图的固定测试集对比",
        f"报告时间：{summary['generated_at']}",
        "状态：三集完整训练、分类评估及 CPU 独立审计均通过。",
        "",
        "AUROC（宏平均）",
        "数据集 | 支持图 原→本次 | 测试图 | 原始 RadZero | 前轮加权 I2CL | 本次加权 I2CL | 相对前轮(百分点) | 相对原始(百分点)",
    ]
    csv_rows = []
    for name in DATASETS:
        row = datasets[name]
        lines.append(" | ".join([
            name, f"{row['previous_support_images']}→{row['support_images']}",
            str(row["test_images"]), percentage(row["baseline_metrics"]["macro_auroc"]),
            percentage(row["previous_i2cl_metrics"]["macro_auroc"]),
            percentage(row["i2cl_metrics"]["macro_auroc"]),
            f"{row['delta_vs_previous_pp']['macro_auroc']:+.2f}",
            f"{row['delta_pp']['macro_auroc']:+.2f}",
        ]))
        item = {key: row[key] for key in (
            "dataset", "previous_support_images", "support_images", "test_images",
            "previous_optimizer_steps", "optimizer_steps", "physical_gpu",
        )}
        for metric in ("auroc", "auprc"):
            for method in ("baseline", "previous_i2cl", "i2cl"):
                item[f"{method}_macro_{metric}"] = row[f"{method}_metrics"][f"macro_{metric}"]
            item[f"delta_vs_previous_{metric}_pp"] = row["delta_vs_previous_pp"][f"macro_{metric}"]
            item[f"delta_vs_baseline_{metric}_pp"] = row["delta_pp"][f"macro_{metric}"]
        csv_rows.append(item)
    lines.extend([
        "", "AUPRC（宏平均，PR 曲线梯形积分）",
        "数据集 | 原始 RadZero | 前轮加权 I2CL | 本次加权 I2CL | 相对前轮(百分点) | 相对原始(百分点)",
    ])
    for name in DATASETS:
        row = datasets[name]
        lines.append(" | ".join([
            name, percentage(row["baseline_metrics"]["macro_auprc"]),
            percentage(row["previous_i2cl_metrics"]["macro_auprc"]),
            percentage(row["i2cl_metrics"]["macro_auprc"]),
            f"{row['delta_vs_previous_pp']['macro_auprc']:+.2f}",
            f"{row['delta_pp']['macro_auprc']:+.2f}",
        ]))
    lines.extend([
        "", "训练与划分：",
        "- 每类从上一轮 5 张 anchor 图中选择 1 张，保留图像完整多标签；同一类别实际正标签数可能大于 1。",
        "- test.csv 字节、图像次序、标签、分组全部保持上一轮不变；未选中的原支持图及其关联组仍不参与测试。",
        "- 新支持图同时用于重建 context 和训练 8 个注入系数，原始 RadZero 全部冻结。",
        "- 新支持集逐类重新统计 pos_weight=负标签数/正标签数；固定学习率 0.01、100 epoch、batch=2 保持不变。",
        "- 固定 epoch 导致优化器更新次数减少，不能将结果变化仅归因于支持图数量：",
    ])
    for name in DATASETS:
        row = datasets[name]
        lines.append(f"  {name}: {row['previous_optimizer_steps']}→{row['optimizer_steps']} 步")
    lines.extend([
        "- 这是 seed=42 的一次试验，尚未多随机种子重复；本轮没有重跑 SIIM/RSNA。",
        "- OpenI 按报告/检查 UID 分组，不能证明跨报告的患者完全去重；ChestXDet10 缺少完整患者 ID。",
        "- 仅报告分类指标，没有本轮定位/分割性能结论。",
        "", f"本轮输出：{run_dir}", f"前轮输出：{parent_run}",
        f"本轮审计：{run_dir / 'audit.json'}",
        f"前轮只读复算：{output / 'parent_cpu_audit.json'}",
    ])
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--parent-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.run_dir, args.parent_run, args.parent_summary, args.output)
    print(f"{summary['status']}: {args.output.resolve() / 'RESULTS.txt'}", flush=True)


if __name__ == "__main__":
    main()
