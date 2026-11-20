#!/usr/bin/env python3
"""Independently audit and summarize all eight-baseline CheXchoNet comparisons.

Reads the completed model outputs identified by the campaign status file. No
models are loaded, no thresholds are fitted, and historical summaries and the
paper draft are never overwritten. Metrics are recomputed from saved scores.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("BiomedCLIP", "BioViL-T", "CheXzero", "MedKLIP", "MAVL", "KAD", "MaCo", "CARZero")
TASKS = ("chexchonet_slvh", "chexchonet_dlv", "chexchonet_composite_slvh_dlv")
TASK_NAMES = dict(zip(TASKS, ("CSLV", "CDLV", "CCOM")))
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
BRANCH_NAMES = dict(zip(BRANCHES, ("原始", "仅查询修正", "仅双视图", "查询修正＋双视图")))
METRICS = ("macro_auroc", "macro_auprc", "macro_average_precision")
DEFAULT_CATALOG = ROOT / "evaluation/chexchonet_20261023/catalog.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def verify_sources(sources, checked):
    for name, expected in sources.items():
        if name in checked:
            if checked[name] != expected:
                raise ValueError(f"Two outputs recorded different versions of {name}")
            continue
        if sha(name) != expected:
            raise ValueError(f"Source SHA256 mismatch: {name}")
        checked[name] = expected


def metrics(labels, scores, classes):
    if labels.shape != scores.shape or scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("Invalid label/score shape")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid saved labels or nonfinite scores")
    per_class = {}
    for index, name in enumerate(classes):
        y, p = labels[:, index], scores[:, index]
        if len(np.unique(y)) != 2:
            raise ValueError("Full evaluation must include both outcomes")
        precision, recall, _ = precision_recall_curve(y, p)
        per_class[name] = {"positive_count": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(auc(recall, precision)), "average_precision": float(average_precision_score(y, p))}
    return {"per_class": per_class, "valid_class_count": len(classes), "class_count": len(classes),
        **{"macro_" + key: float(np.mean([item[key] for item in per_class.values()]))
           for key in ("auroc", "auprc", "average_precision")}}


def compare_metrics(actual, expected, prefix="metrics"):
    """Validate every recomputed metric/count rather than only the table values."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key not in expected:
                raise ValueError(f"Missing reported metric {prefix}.{key}")
            compare_metrics(value, expected[key], prefix + "." + key)
    elif not np.isclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f"Saved/recomputed metric mismatch at {prefix}: {actual} != {expected}")


def audit_parity(report, path="native_parity"):
    if not isinstance(report, dict) or not report:
        raise ValueError(f"Missing parity report: {path}")
    passed_count = 0
    for key, value in report.items():
        if key == "passed":
            if value is not True:
                raise ValueError(f"Native prediction parity failed: {path}")
            passed_count += 1
        elif key == "max_abs_error" and (not np.isfinite(value) or value < 0):
            raise ValueError(f"Invalid parity error: {path}")
        elif isinstance(value, dict):
            passed_count += audit_parity(value, path + "." + key)
    return passed_count


def audit_data(catalog_path, checked):
    catalog = load(catalog_path)
    if catalog.get("status") != "complete" or set(catalog["datasets"]) != set(TASKS):
        raise ValueError("Require completed three-task CheXchoNet catalog")
    checked[str(catalog_path)] = sha(catalog_path)
    prepare_audit_path = catalog_path.parent / "audit.json"
    prepare_audit = load(prepare_audit_path)
    if prepare_audit.get("status") != "passed" or not all(prepare_audit.get(k) for k in (
            "exact_AlphaRAD_saved_image_order_labels_patients", "exact_RadZero_saved_image_order_labels_patients")):
        raise ValueError("Prepared cross-model identity audit is missing or failed")
    checked[str(prepare_audit_path)] = sha(prepare_audit_path)
    verify_sources(prepare_audit["source_sha256"], checked)
    inputs = {}
    support_patients = set()
    shared_order = None
    for name in TASKS:
        entry = catalog["datasets"][name]
        verify_sources({entry["manifest"]: entry["sha256"],
            entry["support_manifest"]: entry["support_manifest_sha256"]}, checked)
        data, supports = load(entry["manifest"]), load(entry["support_manifest"])
        rows, support = data["rows"], supports["support_rows"]
        paths = [r["image"] for r in rows]
        patients = [r["patient_id"] for r in rows]
        labels = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
        if len(rows) != 7407 or len(set(paths)) != 7407 or len(set(patients)) != 2470:
            raise ValueError("Frozen evaluation image/patient count differs")
        if labels.shape != (7407, 1) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Invalid frozen evaluation labels")
        if shared_order is not None and shared_order != list(zip(paths, patients)):
            raise ValueError("Tasks do not share identical ordered evaluation patients/images")
        shared_order = list(zip(paths, patients))
        if len(support) != 10 or len({r["patient_id"] for r in support}) != 10 or sum(r["labels"][0] for r in support) != 5:
            raise ValueError("Expected 5 positive + 5 negative distinct support patients per task")
        support_patients.update(r["patient_id"] for r in support)
        verify_sources(supports["support_sources"], checked)
        verify_sources({r["image"]: r["byte_sha256"] for r in support}, checked)
        inputs[name] = {"paths": paths, "patients": patients, "labels": labels,
            "classes": data["class_names"], "manifest": entry["manifest"],
            "support_manifest": entry["support_manifest"], "support_count": len(support)}
    if len(support_patients) != 30 or support_patients & set(inputs[TASKS[0]]["patients"]):
        raise ValueError("Support union/test patients overlap or support union changed")
    return catalog, inputs, {"test_images_per_task": 7407, "test_patients_per_task": 2470,
        "support_per_task": {"positive": 5, "negative": 5}, "support_union_patients": 30,
        "patient_overlap_count": 0, "excluded_images": 87,
        "exact_prior_AlphaRAD_RadZero_test_order_and_labels": True}


def pattern(record):
    changes = [record["delta_pp"][key] for key in ("macro_auroc", "macro_auprc")]
    return "两项上升" if min(changes) > 0 else "两项下降" if max(changes) < 0 else "混合或持平"


def score_percent(value):
    return f"{100 * value:.2f}"


def signed(value):
    return f"{value:+.5f}" if 0 < abs(value) < .005 else f"{value:+.2f}"


def csv_write(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run):
    state = load(run / "status.json")
    catalog_path = Path(state.get("catalog", DEFAULT_CATALOG)).resolve()
    if state.get("catalog_sha256") and sha(catalog_path) != state["catalog_sha256"]:
        raise ValueError("Campaign catalog SHA256 changed after launch")
    checked = {str(Path(__file__).resolve()): sha(__file__)}
    catalog, inputs, data_audit = audit_data(catalog_path, checked)
    if set(state["models"]) != set(MODELS):
        raise ValueError("Campaign must contain all eight requested baselines")
    records, audits = [], []
    parity_checks = 0
    for model in MODELS:
        output = Path(state["models"][model]["full_output"])
        if not output.is_absolute():
            output = run / output
        worker_state = load(output / "status.json")
        if worker_state.get("status") != "complete" or worker_state.get("smoke") or worker_state.get("changed_sources"):
            raise ValueError(f"{model} full worker is not complete and source-stable")
        if worker_state.get("outcomes") != dict.fromkeys(TASKS, "complete"):
            raise ValueError(f"{model} did not complete all three tasks")
        verify_sources(load(output / "source_sha256.json"), checked)
        for name in TASKS:
            expected = inputs[name]
            directory = output / name
            result_path = directory / "result.json"
            prediction_path = directory / "predictions.npz"
            result = load(result_path)
            if (result.get("status") != "complete" or result.get("smoke") or result.get("rows") != 7407
                    or result.get("model") != model or result.get("dataset") != name
                    or result.get("support_count") != 10 or result.get("eta") != .3
                    or result.get("test_labels_used_for_query_fit") is not False
                    or result.get("no_gradient_training") is not True):
                raise ValueError(f"{model}/{name} run protocol mismatch")
            if result["class_names"] != expected["classes"]:
                raise ValueError(f"{model}/{name} class order differs")
            verify_sources(result["source_sha256"], checked)
            parity = result["native_parity"]
            if not parity.get("native", {}).get("passed"):
                raise ValueError(f"{model}/{name} missing native parity check")
            if model in {"BiomedCLIP", "BioViL-T", "CheXzero", "MaCo"} and not parity.get("joint_vs_individual", {}).get("passed"):
                raise ValueError(f"{model}/{name} missing independent-query parity")
            parity_checks += audit_parity(parity)
            native = result["original_adapter_metadata"]
            if model in {"MedKLIP", "MAVL"}:
                integrity = native.get("extension_integrity", {})
                if not integrity.get("all_checkpoint_state_values_unchanged") or integrity.get("state_sha256_before") != integrity.get("state_sha256_after"):
                    raise ValueError(f"{model}: query overlay did not preserve checkpoint weights")
                if native.get("query_extension", {}).get("exact_author_prompts") is not False:
                    raise ValueError(f"{model}: reconstructed prompts not explicitly disclosed")
            branches = {}
            with np.load(prediction_path, allow_pickle=False) as saved:
                if saved["image_paths"].tolist() != expected["paths"] or saved["patient_ids"].tolist() != expected["patients"]:
                    raise ValueError(f"{model}/{name} saved image/patient order differs")
                if saved["class_names"].tolist() != expected["classes"] or not np.array_equal(saved["labels"], expected["labels"]):
                    raise ValueError(f"{model}/{name} saved classes/labels differ")
                for branch in BRANCHES:
                    value = metrics(saved["labels"], saved[branch], expected["classes"])
                    compare_metrics(value, result["branches"][branch], f"{model}/{name}/{branch}")
                    branches[branch] = value
            delta = {key: 100 * (branches["tta_center_refined"][key] - branches["original"][key]) for key in METRICS}
            compare_metrics(delta, result["delta_pp"], f"{model}/{name}/delta_pp")
            records.append({"status": "complete", "model": model, "dataset": name, "task": "classification",
                "task_display": TASK_NAMES[name], "rows": 7407, "patients": 2470, "support_count": 10,
                "branches": branches, "delta_pp": delta, "result": str(result_path),
                "result_sha256": sha(result_path), "predictions": str(prediction_path),
                "predictions_sha256": sha(prediction_path), "native_score_kind": native.get("score_kind", "see original_adapter_metadata"),
                "native_parity": parity, "original_adapter_metadata": native,
                "query_adapter_metadata": result["metadata"], "protocol": result["protocol"]})
            audits.append({"model": model, "dataset": name, "status": "passed", "rows": 7407,
                "all_four_branches_metrics_recomputed": True, "labels_paths_patients_exact": True})
    counts = {"models": 8, "tasks_per_model": 3, "paired_comparisons": len(records),
        "prediction_branches": len(records) * len(BRANCHES),
        "both_up": sum(pattern(r) == "两项上升" for r in records),
        "both_down": sum(pattern(r) == "两项下降" for r in records),
        "mixed_or_tied": sum(pattern(r) == "混合或持平" for r in records)}
    model_summary = []
    for model in MODELS:
        selected = [r for r in records if r["model"] == model]
        branches = {branch: {key: float(np.mean([r["branches"][branch][key] for r in selected])) for key in METRICS} for branch in BRANCHES}
        model_summary.append({"model": model, "task_count": 3, "branches": branches,
            "delta_pp": {key: 100 * (branches["tta_center_refined"][key] - branches["original"][key]) for key in METRICS}})
    audit = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_comparison_atol": 1e-10, "metric_comparison_rtol": 0,
        "metric_values_recomputed": len(records) * 4 * 3,
        "native_and_joint_parity_pass_records": parity_checks, "data": data_audit,
        "records": audits, "verified_sources_sha256": checked}
    warnings = [
        "Custom 7494-image subset is not the AlphaRAD paper split. Compare only matched 7407-image arms.",
        "Three cardiac tasks share the same images/patients; their mean is a three-task mean, not a three-dataset mean.",
        "Every baseline retains its native score domain. Seven models average probabilities; CARZero averages native logits. This is not universally logit averaging.",
        "MedKLIP and MAVL require fixed experimental unseen-query resources; their original branch here means original checkpoint and scoring with this same new task-query overlay.",
        "MedKLIP appends three fixed statements to its original 75 descriptions; MAVL repeats each statement in all eight concept slots and its description slot. Author CheXchoNet description/concept resources were unavailable.",
        "No test-driven model/branch/prompt selection. Fixed eta=.3, original-view supports, native-scale query refinement and center80 score averaging.",
        "AUPRC is the trapezoidal PR-curve area. Average precision (AP) is saved separately and must not be relabeled AUPRC.",
        "Labeled-support adaptation and a pure zero-shot model have different information budgets; frozen weights do not imply no labeled information.",
        "Results are descriptive point estimates without a statistical-significance claim; pretraining source overlap is not established."]
    payload = {"status": "complete", "scope": catalog["scope"], "catalog": str(catalog_path),
        "catalog_sha256": sha(catalog_path), "counts": counts, "records": records,
        "model_three_task_summary": model_summary, "protocol_notes": warnings,
        "audit": str(run / "FINAL_AUDIT.json"), "generated_at_utc": audit["created_at_utc"]}
    comparison_rows, four_arm_rows = [], []
    for record in records:
        row = {"model": record["model"], "dataset": record["dataset"], "task": record["task_display"],
            "test_images": 7407, "test_patients": 2470, "support_images": 10,
            "status": "complete", "change": pattern(record), "native_score_kind": record["native_score_kind"]}
        for key in METRICS:
            row.update({key + "_original_percent": 100 * record["branches"]["original"][key],
                key + "_refined_percent": 100 * record["branches"]["tta_center_refined"][key],
                key + "_delta_pp": record["delta_pp"][key]})
        comparison_rows.append(row)
        for branch in BRANCHES:
            four_arm_rows.append({"model": record["model"], "dataset": record["dataset"], "task": record["task_display"],
                "branch": branch, "test_images": 7407, "test_patients": 2470, "support_images_for_adaptation": 10,
                **{key + "_percent": 100 * record["branches"][branch][key] for key in METRICS},
                "native_score_kind": record["native_score_kind"]})
    lines = ["8个baseline：CheXchoNet 三任务，原始与查询修正＋双视图对照", "",
        "已完成8模型×3任务=24组对照；每组4臂，共96项分支结果。全部通过保存预测重算及数据/代码审计。",
        "CSLV=重度左心室肥厚；CDLV=左心室扩张；CCOM=上述两者任一阳性。",
        "支持集与之前AlphaRAD/RadZero完全相同：每任务5正＋5负；30支持患者的87张图被排除。",
        "各任务所有模型和分支统一测试7,407张、2,470患者；CSLV/CDLV/CCOM分别567/315/828张阳性。",
        "原始7,494张是自定义抽取，不是论文原始划分；不能将本表与论文数值当成严格同集复现对比。",
        "固定η=0.3；网络权重冻结；支持集原视图计算查询方向；测试使用原视图和中心80%裁剪视图。",
        "双视图平均保持各模型原生评分域：7个模型平均概率；CARZero平均原生logit。不是所有模型统一平均logit。",
        "指标单位为%；涨跌单位为百分点，按未四舍五入值计算。AUPRC=PR曲线梯形面积，AP另列。", "",
        "一、全部24组公平对照（原始→完整方案）",
        "| 模型 | 任务 | 原始AUROC | 完整方案AUROC | AUROC涨跌 | 原始AUPRC | 完整方案AUPRC | AUPRC涨跌 | 变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for record in records:
        cells = []
        for key in METRICS[:2]:
            cells += [score_percent(record["branches"]["original"][key]), score_percent(record["branches"]["tta_center_refined"][key]), signed(record["delta_pp"][key])]
        lines.append(f"| {record['model']} | {record['task_display']} | " + " | ".join(cells) + f" | {pattern(record)} |")
    lines += ["", f"两项都升：{counts['both_up']}/24；两项都降：{counts['both_down']}/24；混合或持平：{counts['mixed_or_tied']}/24。",
        "仅描述点估计涨跌，不代表统计显著。", "", "二、每模型三任务等权平均（同一数据集，不是三个独立数据集）",
        "| 模型 | 原始AUROC | 完整方案AUROC | AUROC涨跌 | 原始AUPRC | 完整方案AUPRC | AUPRC涨跌 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in model_summary:
        cells = []
        for key in METRICS[:2]:
            cells += [score_percent(row["branches"]["original"][key]), score_percent(row["branches"]["tta_center_refined"][key]), signed(row["delta_pp"][key])]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    lines += ["", "三、四臂消融明细（AUROC / AUPRC / AP）",
        "四臂固定为原始、仅查询修正、仅双视图、查询修正＋双视图；没有逐任务挑选最佳分支。",
        "| 模型 | 任务 | 分支 | AUROC | AUPRC | AP |", "|---|---|---|---:|---:|---:|"]
    for record in records:
        for branch in BRANCHES:
            cells = [score_percent(record["branches"][branch][key]) for key in METRICS]
            lines.append(f"| {record['model']} | {record['task_display']} | {BRANCH_NAMES[branch]} | " + " | ".join(cells) + " |")
    lines += ["", "四、MedKLIP/MAVL新增查询资源与比较范围",
        "这两者的作者资源没有提供本次三个任务的完整描述/概念，因此增加了明确标记的实验性文本查询扩展。",
        "MedKLIP：保留原75疾病查询，再加入3条固定心脏疾病描述，使用作者文本编码器及共享分类头。",
        "MAVL：保留原75×9查询上下文；每个新任务的固定描述同时填入8个概念槽和主描述槽，沿用原46轮checkpoint及9槽共享头。",
        "MAVL这部分没有作者提供的丰富视觉概念，属于重复描述扩展，不能称为作者CheXchoNet协议的严格复现。",
        "这两模型的原始臂和改进臂使用完全相同的新查询资源及checkpoint；原始臂不使用支持集修正，改进臂才使用。",
        "没有把这些任务替换成心脏增大/心影增大等代理标签，也没有改动checkpoint参数。",
        "其他模型沿用各自既有文本模板和预处理，使用明确心脏疾病语义；作者精确CheXchoNet提示词未获核实。", "",
        "五、审计与文件",
        "所有预测逐张核对图片顺序、患者和标签；保持与之前AlphaRAD/RadZero相同7407张测试及固定支持集。",
        "从保存的96组分支预测独立重算AUROC、梯形AUPRC、AP，与运行时结果在1e-10绝对误差内一致。",
        "核对源文件SHA256、原生baseline一致性，以及共享前向与逐任务查询一致性；没有重新训练。",
        "本方法使用带标签支持集，属于冻结模型的少样本适配；与纯零样本基线的信息预算不同。",
        f"FINAL_AUDIT.json：{run / 'FINAL_AUDIT.json'}", f"完整JSON：{run / 'RESULTS.json'}",
        f"原始/完整方案CSV：{run / 'COMPARISON.csv'}", f"四臂CSV：{run / 'FOUR_ARM_RESULTS.csv'}",
        f"冻结数据catalog：{catalog_path}"]
    write(run / "campaign_status_at_summary.json", state)
    write(run / "FINAL_AUDIT.json", audit)
    write(run / "RESULTS.json", payload)
    csv_write(run / "COMPARISON.csv", comparison_rows)
    csv_write(run / "FOUR_ARM_RESULTS.csv", four_arm_rows)
    (run / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    return {"status": "complete", "counts": counts, "results": str(run / "RESULTS.txt"), "audit": str(run / "FINAL_AUDIT.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    try:
        report = summarize(run)
    except Exception as error:
        if run.is_dir():
            write(run / "FINAL_AUDIT.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
