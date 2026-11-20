#!/usr/bin/env python3
"""Compare audited weighted/unweighted I2CL runs on identical supports and tests.

Historical artifacts are read only. This comparison independently recomputes
both sets of classification metrics and checks that the experimental settings,
except for the requested positive-class loss weights, remain the same.
"""

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from audit_i2cl import (
    SUPPORT_WEIGHTED_LOSS, UNWEIGHTED_LOSS, fingerprint, identity,
    independent_metrics, read_json, read_predictions, require, sha256,
    verify_metrics, write_json,
)
from batch_data import DATASETS
from i2cl_data import load_spec


def normalized_config(config):
    """Discard only execution provenance and the loss under comparison."""
    value = copy.deepcopy(config)
    for key in ("created_at", "cuda_visible_devices", "gpu_name", "fingerprint",
                "source_sha256", "split_sha256", "loss", "loss_weighting"):
        value.pop(key, None)
    for key in ("split_dir", "output", "baseline_dir"):
        value["arguments"].pop(key, None)
    # The merged root has a new root manifest, but dataset files are unchanged.
    for key in ("split_manifest", "root_split_fingerprint"):
        value["split_metadata"].pop(key, None)
    return value


def file_tree(path):
    return {str(item.relative_to(path)): sha256(item)
            for item in path.rglob("*") if item.is_file()}


def compare_dataset(name, previous, current, old_summary_path):
    old_dir, new_dir = Path(previous["run_directory"]), Path(current["run_directory"])
    old_config = read_json(old_dir / "run_config.json")
    new_config = read_json(new_dir / "run_config.json")
    require(old_config["loss"] == UNWEIGHTED_LOSS, f"{name}: previous loss is not unweighted")
    require(new_config["loss"] == SUPPORT_WEIGHTED_LOSS, f"{name}: new loss is not pos_weight BCE")
    require(normalized_config(old_config) == normalized_config(new_config),
            f"{name}: experimental configuration changed beyond loss and execution paths")
    source_changes = {}
    require(old_config["source_sha256"].keys() == new_config["source_sha256"].keys(),
            f"{name}: training source file set differs")
    for path, old_hash in old_config["source_sha256"].items():
        new_hash = new_config["source_sha256"][path]
        if old_hash != new_hash:
            source_changes[Path(path).name] = {"previous": old_hash, "current": new_hash}
        snapshot = old_summary_path.parent / "source_snapshot/local_setup" / Path(path).name
        require(snapshot.is_file() and sha256(snapshot) == old_hash,
                f"{name}: archived original source checksum differs for {path}")
    require(set(source_changes) == {"train_i2cl.py"},
            f"{name}: unexpected core implementation changes: {source_changes.keys()}")

    old_split, new_split = Path(old_config["arguments"]["split_dir"]), Path(new_config["arguments"]["split_dir"])
    require(file_tree(old_split / name) == file_tree(new_split / name),
            f"{name}: per-dataset split directory differs")
    for split in ("support", "test"):
        old_spec, new_spec = load_spec(old_split, name, split), load_spec(new_split, name, split)
        require(identity(old_spec) == identity(new_spec), f"{name}: {split} identity differs")
        require(fingerprint(identity(old_spec)) == old_config[f"{split}_identity"]
                == new_config[f"{split}_identity"],
                f"{name}: recorded {split} identity differs from actual samples")
    test = load_spec(new_split, name, "test")

    measured = {}
    baseline_logits = {}
    adapted_logits = {}
    training_records = {}
    for label, directory, row, config in (
            ("previous", old_dir, previous, old_config),
            ("current", new_dir, current, new_config)):
        result = read_json(directory / "result.json")
        training_records[label] = result["training"]
        require(result["status"] == "complete" and not result["is_smoke_test"],
                f"{name}: incomplete or smoke {label} run")
        require(fingerprint({k: v for k, v in config.items() if k != "fingerprint"})
                == config["fingerprint"] == result["run_fingerprint"],
                f"{name}: invalid {label} configuration fingerprint")
        require(sha256(directory / "i2cl.pt") == result["checkpoint_sha256"],
                f"{name}: {label} checkpoint checksum differs")
        measured[label] = {}
        for method, filename, hash_key in (
                ("i2cl", "predictions.csv", "predictions_sha256"),
                ("baseline", "baseline_predictions.csv", "baseline_predictions_sha256")):
            require(sha256(directory / filename) == result[hash_key],
                    f"{name}: {label} {method} prediction checksum differs")
            logits = read_predictions(directory / filename, test)
            metrics = independent_metrics(test["labels"], logits, test["class_names"])
            verify_metrics(row[f"{method}_metrics"], metrics, f"{name}/{label}/{method}")
            verify_metrics(result["metrics" if method == "i2cl" else "baseline_metrics"],
                           metrics, f"{name}/{label}/{method}/result")
            measured[label][method] = metrics
            if method == "baseline":
                baseline_logits[label] = logits
            else:
                adapted_logits[label] = logits
    baseline_error = float(np.max(np.abs(baseline_logits["previous"] - baseline_logits["current"])))
    require(np.allclose(baseline_logits["previous"], baseline_logits["current"],
                        rtol=1e-6, atol=3e-5),
            f"{name}: unchanged-model baseline logits differ beyond numerical tolerance")
    require(previous["context_tensor_sha256"] == current["context_tensor_sha256"],
            f"{name}: context changed although the support and context calculation should be unchanged")
    require(previous["base_tensor_digest"] == current["base_tensor_digest"],
            f"{name}: base-model tensor digest changed")

    result = {
        "dataset": name,
        "support_images": current["support_images"],
        "test_images": current["test_images"],
        "support_identity": new_config["support_identity"],
        "test_identity": new_config["test_identity"],
        "previous_run": str(old_dir),
        "current_run": str(new_dir),
        "same_split_files_and_sample_order": True,
        "same_prompts_labels_checkpoint_and_nonloss_config": True,
        "same_context_tensor_digest": True,
        "only_changed_training_source": source_changes,
        "baseline_max_abs_logit_difference": baseline_error,
        "baseline_logits_bitwise_equal": bool(np.array_equal(
            baseline_logits["previous"], baseline_logits["current"])),
        "class_names": new_config["class_names"],
        "loss_weighting": new_config["loss_weighting"],
        "metrics": measured,
        "delta_new_vs_previous_pp": {},
        "delta_new_vs_baseline_pp": {},
    }
    old_coefficients = np.asarray(training_records["previous"]["final_coefficients"])
    new_coefficients = np.asarray(training_records["current"]["final_coefficients"])
    result["coefficient_comparison"] = {
        "previous": old_coefficients.tolist(),
        "current": new_coefficients.tolist(),
        "max_abs_difference": float(np.max(np.abs(old_coefficients - new_coefficients))),
        "bitwise_equal": bool(np.array_equal(old_coefficients, new_coefficients)),
    }
    if all(value == 1.0 for value in new_config["loss_weighting"]["pos_weight"]):
        result["unit_weight_equivalence"] = {
            "pos_weight_all_one": True,
            "mathematically_same_loss": True,
            "adapted_predictions_max_abs_logit_difference": float(np.max(np.abs(
                adapted_logits["previous"] - adapted_logits["current"]))),
            "adapted_predictions_bitwise_equal": bool(np.array_equal(
                adapted_logits["previous"], adapted_logits["current"])),
            "numerical_comparison_tolerance": {"rtol": 1e-6, "atol": 3e-5},
            "adapted_predictions_within_tolerance": bool(np.allclose(
                adapted_logits["previous"], adapted_logits["current"],
                rtol=1e-6, atol=3e-5)),
        }
    for key in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc"):
        old = measured["previous"]["i2cl"][key]
        new = measured["current"]["i2cl"][key]
        base = measured["current"]["baseline"][key]
        result["delta_new_vs_previous_pp"][key] = None if old is None or new is None else (new - old) * 100
        result["delta_new_vs_baseline_pp"][key] = None if base is None or new is None else (new - base) * 100
    return result


def compare(previous_path, current_path, output):
    previous_path, current_path = previous_path.resolve(), current_path.resolve()
    previous, current = read_json(previous_path), read_json(current_path)
    for label, summary in (("previous", previous), ("current", current)):
        require(summary["status"] == "complete_and_audited",
                f"{label} summary is not complete and audited")
        require(set(summary["datasets"]) == set(DATASETS), f"{label} missing datasets")
        for report in summary["audit_reports"]:
            require(sha256(report["path"]) == report["sha256"],
                    f"{label} audit report was modified")
            require(read_json(report["path"])["status"] == "passed",
                    f"{label} audit did not pass")
    rows = {name: compare_dataset(name, previous["datasets"][name],
                                 current["datasets"][name], previous_path)
            for name in DATASETS}
    report = {
        "status": "passed",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "previous_summary": {"path": str(previous_path), "sha256": sha256(previous_path)},
        "current_summary": {"path": str(current_path), "sha256": sha256(current_path)},
        "comparison_code_sha256": sha256(__file__),
        "metric_units": "fractions; deltas in percentage points",
        "auprc_method": "trapezoidal PR curve area, not average precision",
        "scope": "Same support/test rows, labels, groups, prompts, base model, context, and "
                 "non-loss training settings. CPU metrics independently recomputed for old/new "
                 "predictions. Previous artifacts are not rewritten.",
        "datasets": rows,
    }
    output = output.resolve()
    require(output != previous_path.parent, "Refusing to write comparison into historical run")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "COMPARISON.json", report)
    lines = [
        "RadZero_i2cl：同一支持集/测试集上的 pos_weight BCE 对照",
        f"生成时间：{report['generated_at']}",
        "",
        "AUROC（宏平均）：",
        "数据集 | 原始 RadZero | 旧版未加权 I2CL | 新版 pos_weight I2CL | 相对旧版变化(百分点)",
    ]
    csv_rows = []
    for metric in ("macro_auroc", "macro_auprc"):
        if metric == "macro_auprc":
            lines += ["", "AUPRC（宏平均，PR 曲线梯形积分）：",
                      "数据集 | 原始 RadZero | 旧版未加权 I2CL | 新版 pos_weight I2CL | 相对旧版变化(百分点)"]
        for name, row in rows.items():
            base = row["metrics"]["current"]["baseline"][metric]
            old = row["metrics"]["previous"]["i2cl"][metric]
            new = row["metrics"]["current"]["i2cl"][metric]
            delta = row["delta_new_vs_previous_pp"][metric]
            shown_delta = f"{delta:+.4f}" if 0 < abs(delta) < .01 else f"{delta:+.2f}"
            lines.append(f"{name} | {100*base:.2f}% | {100*old:.2f}% | {100*new:.2f}% | {shown_delta}")
    for name, row in rows.items():
        flat = {"dataset": name, "support_images": row["support_images"],
                "test_images": row["test_images"]}
        for metric in ("macro_auroc", "macro_auprc", "micro_auroc", "micro_auprc"):
            flat[f"baseline_{metric}"] = row["metrics"]["current"]["baseline"][metric]
            flat[f"previous_{metric}"] = row["metrics"]["previous"]["i2cl"][metric]
            flat[f"current_{metric}"] = row["metrics"]["current"]["i2cl"][metric]
            flat[f"delta_new_vs_previous_{metric}_pp"] = row["delta_new_vs_previous_pp"][metric]
        csv_rows.append(flat)
    lines += [
        "",
        "对照核验：五集支持/测试文件逐字节相同，图像顺序、标签、分组、文本提示、",
        "原始模型、未加权上下文完全一致；seed=42、100 轮、batch=2、固定 lr=0.01 保持不变。",
        "仅训练 loss 改为按整个支持集逐类统计 negatives/positives 的 pos_weight BCE。",
        "每个 batch 不单独重新计算权重；上下文仍为未加权均值。",
        "原始 RadZero、旧版未加权 I2CL、新版加权 I2CL 均使用同一测试列表。",
        "SIIM/RSNA 支持集 5 阳性+5 阴性，pos_weight=1，数学上与普通 BCE 等价。",
        "",
        "这是单个 seed 的本地分类对照；未报告定位或 Dice，不代表论文完整测试协议。",
        "OpenI 为报告级隔离，无法核实跨报告患者身份；ChestXDet10 无完整患者 ID。",
        f"旧版结果：{previous_path}",
        f"新版结果：{current_path}",
        "逐类指标、固定权重、配置/数据一致性与数值差异见 COMPARISON.json。",
    ]
    (output / "COMPARISON.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "COMPARISON.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-summary", type=Path, required=True)
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.previous_summary, args.current_summary, args.output)
    print(json.dumps({"status": report["status"], "datasets": list(report["datasets"]),
                      "report": str(args.output.resolve() / "COMPARISON.txt")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
