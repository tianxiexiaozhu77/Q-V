#!/usr/bin/env python3
"""Audit and combine completed local RadZero_i2cl classification experiments."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from audit_i2cl import (
    SUPPORT_WEIGHTED_LOSS, UNWEIGHTED_LOSS, audit, require, sha256, write_json,
)
from batch_data import DATASETS


def percentage(value):
    return "未定义" if value is None else f"{100 * value:.2f}%"


def common_loss(datasets):
    losses = {row["loss"] for row in datasets.values()}
    require(len(losses) == 1,
            "Cannot combine different training losses into one five-dataset experiment")
    loss = next(iter(losses))
    require(loss in (UNWEIGHTED_LOSS, SUPPORT_WEIGHTED_LOSS),
            f"Unsupported training loss: {loss}")
    return loss


def common_regularization(datasets):
    settings = {
        json.dumps(row.get("regularization"), sort_keys=True)
        for row in datasets.values()
    }
    require(len(settings) == 1,
            "Cannot combine different coefficient regularization settings into one experiment")
    return json.loads(next(iter(settings)))


def summarize(run_dirs, output):
    directories = [Path(directory).expanduser().resolve() for directory in run_dirs]
    require(len(set(directories)) == len(directories), "Duplicate run directories")
    # Do not represent a still-running experiment as a completed result.
    for directory in directories:
        launcher = json.loads((directory / "launcher.json").read_text())
        require(launcher["status"] == "complete", f"Run has not completed: {directory}")
        require(not launcher["is_smoke"], f"Smoke results cannot enter this report: {directory}")
    datasets, audit_files = {}, []
    for directory in directories:
        report = audit(directory)
        audit_file = directory / "audit.json"
        audit_files.append({"path": str(audit_file), "sha256": sha256(audit_file)})
        for name, row in report["datasets"].items():
            require(name not in datasets, f"Multiple completed results for {name}; select one run")
            datasets[name] = dict(row, run_directory=str(directory / name))
    require(set(datasets) == set(DATASETS),
            f"Expected the five public datasets; received {sorted(datasets)}")
    loss = common_loss(datasets)
    regularization = common_regularization(datasets)
    weighted = loss == SUPPORT_WEIGHTED_LOSS
    for name, row in datasets.items():
        require(row["epochs"] == 100 and row["grad_batch_size"] == 2,
                f"{name}: original epoch/batch settings changed")
        require(row["learning_rate"] == .01 and row["weight_decay"] == .001 and
                row["noise_scale"] == .001, f"{name}: original optimizer/noise settings changed")
        require(row["sampling"] ==
                "Each epoch shuffles and consumes the same fixed support set exactly once",
                f"{name}: original traversal changed")
        if weighted:
            require(row.get("loss_weighting") is not None,
                    f"{name}: support loss weights were not audited")
        require(row["same_support_context_and_calibration"] and row["trainable_scalars"] == 8,
                f"{name}: support/model protocol changed")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete_and_audited",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "project": "RadZero_i2cl",
        "run_directories": [str(directory) for directory in directories],
        "audit_reports": audit_files,
        "summary_code_sha256": sha256(__file__),
        "metric_units": "AUROC/AUPRC fractions; delta_pp in percentage points",
        "auprc_method": "Trapezoidal area under precision-recall curve; not average precision",
        "training": {
            "epochs": 100, "gradient_batch_size": 2,
            "optimizer": "AdamW", "constant_learning_rate": .01,
            "weight_decay": .001, "activation_noise_scale": .001,
            "trainable_scalars": 8, "context": "unweighted CLS mean",
            "loss": loss,
            "objective": next(iter(datasets.values())).get("objective", loss),
            "regularization": regularization,
            "loss_weighting_by_dataset": {
                name: row.get("loss_weighting") for name, row in datasets.items()
            },
            "sampling": "shuffle once per epoch; consume all supports including final short batch",
            "checkpoint_selection": "final fixed epoch; no validation/test checkpoint selection",
        },
        "datasets": datasets,
        "limitations": [
            "Only classification AUROC/AUPRC are evaluated; no localization or Dice claim.",
            "All results use a single seed (42) and dataset-specific support calibration.",
            "Support images are excluded from the corresponding holdout together with their groups.",
            "SIIM uses the explicitly authorized released-test few-shot group holdout because "
            "an independent local training pool is unavailable.",
            "The different test/support protocols prevent direct comparison with the earlier "
            "weighted, warmup/cosine RadZero implementation or the paper's full-test zero-shot scores.",
            "OpenI report grouping does not establish patient identity across reports; "
            "ChestXDet10 has no complete patient identifier mapping.",
        ],
    }
    write_json(output / "summary.json", summary)
    lines = [
        "RadZero_i2cl：五个数据集训练、分类评估与 CPU 审计全部完成",
        f"报告时间：{summary['generated_at']}",
        "",
        "数据集 | 支持图/测试图 | 原始 AUROC | I2CL AUROC | AUROC 变化(百分点) | 原始 AUPRC | I2CL AUPRC",
    ]
    csv_rows = []
    for name in DATASETS:
        row = datasets[name]
        baseline, adapted = row["baseline_metrics"], row["i2cl_metrics"]
        change = row["delta_pp"]["macro_auroc"]
        lines.append(" | ".join([
            name, f"{row['support_images']}/{row['test_images']}",
            percentage(baseline["macro_auroc"]), percentage(adapted["macro_auroc"]),
            "未定义" if change is None else f"{change:+.2f}",
            percentage(baseline["macro_auprc"]), percentage(adapted["macro_auprc"]),
        ]))
        csv_rows.append({
            "dataset": name, "support_images": row["support_images"],
            "test_images": row["test_images"], "physical_gpu": row["physical_gpu"],
            "baseline_macro_auroc": baseline["macro_auroc"],
            "i2cl_macro_auroc": adapted["macro_auroc"],
            "delta_auroc_pp": change,
            "baseline_macro_auprc": baseline["macro_auprc"],
            "i2cl_macro_auprc": adapted["macro_auprc"],
            "delta_auprc_pp": row["delta_pp"]["macro_auprc"],
            "protocol": row["protocol"], "run_directory": row["run_directory"],
            "loss": row["loss"],
            "loss_weighting_json": json.dumps(row.get("loss_weighting"), ensure_ascii=False),
            "regularization_json": json.dumps(row.get("regularization"), ensure_ascii=False),
        })
    lines.extend([
        "",
        "训练设置：100 轮，batch=2，AdamW 固定学习率 0.01，weight_decay=0.001，",
        "每轮随机打乱并完整遍历支持集；上下文使用未加权均值；只训练 8 个注入系数。",
        ("分类使用 BCEWithLogitsLoss，逐类 pos_weight=支持集阴性数/阳性数，reduction=mean。"
         if weighted else "分类使用普通未加权 BCEWithLogitsLoss。"),
        ("注入系数恒等点正则："
         f"lambda_reg={regularization['lambda']:g}，"
         "总损失 = 分类 BCE + lambda_reg × [mean(alpha²) + mean((beta−1)²)]。"
         if regularization is not None else "未使用注入系数恒等点正则。"),
        "每个疾病类别 5 张；二分类数据集为 5 阳性 + 5 阴性；向量生成与系数训练使用完全相同支持集。",
        "未启用上一轮的背景比例加权或学习率预热/余弦调度。",
        "",
        "SIIM 例外：从本地已有 test 中抽取 5 阳性 + 5 阴性作为支持集，",
        "按患者、检查及文件重复分组排除关联图像后评估，已逐个 DICOM 核对患者与检查 ID。",
        "原本的独立 train 数据不可用；此次结果不是论文完整 test 上的零样本结果。",
        "",
        "基线与 I2CL 在完全相同的测试图像和标签上评估。AUROC/AUPRC 为宏平均，",
        "AUPRC 为 PR 曲线梯形积分；本次没有评估定位命中率或分割 Dice。",
        "CPU 审计复算全部分类指标，并检查源文件、划分、模型、checkpoint、预测与缓存哈希，",
        "以及同一支持集、组隔离、8 系数训练、原模型冻结、checkpoint 重载记录。",
        "冻结和梯度记录属于运行时检查，CPU 审计不重复 GPU 前向或反向计算。",
        "",
        "限制：单个 seed=42；OpenI 仅可核实报告级分组，ChestXDet10 缺完整患者 ID。",
        "本次支持/测试划分与上一轮 RadZero 实现不同，不能将指标差异直接归因于代码版本。",
        "",
        "审计文件：",
        *[item["path"] for item in audit_files],
    ])
    if weighted:
        lines.extend(["", "各数据集固定分类权重（只统计训练支持集；按类别顺序）："])
        for name in DATASETS:
            row = datasets[name]
            weights = row["loss_weighting"]
            lines.append(f"{name}: " + json.dumps({
                "class_names": row["class_names"],
                "positive_counts": weights["positive_counts"],
                "negative_counts": weights["negative_counts"],
                "pos_weight": weights["pos_weight"],
            }, ensure_ascii=False))
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize([path for group in args.run_dir for path in group], args.output)
    print(json.dumps({
        "status": summary["status"], "datasets": list(summary["datasets"]),
        "report": str(args.output.resolve() / "RESULTS.txt"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
