#!/usr/bin/env python3
"""Independently verify and summarize the four frozen CheXchoNet jobs.

Uses saved predictions, not rounded console/table values. A full-set baseline
is matched by exact image path to the held-out baseline and checked at 2e-5
absolute logit tolerance. Metrics retain each model's established score type:
AlphaRAD native sigmoid probabilities and RadZero native logits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

TASKS = ["slvh", "dlv", "composite_slvh_dlv"]
DISPLAY = {"slvh": "CSLV / 严重左心室肥厚", "dlv": "CDLV / 左心室扩张",
           "composite_slvh_dlv": "CCOM / 两者任一阳性"}
ARMS = ["original", "uniform_eta03", "tta_center_original", "tta_center_refined"]
ARM_NAMES = {"original": "原始模型", "uniform_eta03": "仅查询修正",
             "tta_center_original": "仅双视图平均", "tta_center_refined": "查询修正＋双视图平均"}
KEYS = ("auroc", "auprc", "average_precision")


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metrics(labels, scores):
    if labels.shape != scores.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("Require aligned [N,3] labels/scores")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid label or score")
    per = {}
    for j, task in enumerate(TASKS):
        y, p = labels[:, j], scores[:, j]
        if len(np.unique(y)) != 2:
            raise ValueError(f"No valid binary metric for {task}")
        precision, recall, _ = precision_recall_curve(y, p)
        per[task] = {"auroc": float(roc_auc_score(y, p)),
                     "auprc": float(auc(recall, precision)),
                     "average_precision": float(average_precision_score(y, p)),
                     "positive": int(y.sum()), "negative": int(len(y) - y.sum())}
    return {"per_task": per, **{"macro_" + key: float(np.mean([per[t][key] for t in TASKS])) for key in KEYS}}


def assert_metrics(expected, computed, *, alpha):
    per = expected["per_task"] if alpha else expected["per_class"]
    keys = KEYS if alpha else ("auroc", "auprc")
    for task in TASKS:
        for key in keys:
            np.testing.assert_allclose(per[task][key], computed["per_task"][task][key], atol=1e-12, rtol=0,
                                       err_msg=f"Saved metric disagrees with predictions: {task}/{key}")
    for key in keys:
        np.testing.assert_allclose(expected["macro_" + key], computed["macro_" + key], atol=1e-12, rtol=0)


def check_result(path, mode):
    result = read_json(path)
    if result["status"] != "complete" or result["mode"] != mode:
        raise ValueError(f"Incomplete or wrong job mode: {path}")
    if result.get("smoke_batches") or result.get("is_smoke"):
        raise ValueError(f"Smoke data cannot appear in a formal summary: {path}")
    if result["tasks"] != TASKS:
        raise ValueError(f"Task order mismatch: {path}")
    return result


def validate_arrays(paths, patients, labels, logits):
    if len(paths) != len(set(paths)) or len(patients) != len(paths):
        raise ValueError("Duplicate or misaligned image/patient identities")
    if labels.shape != (len(paths), 3) or logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Invalid prediction shape or values")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Nonbinary labels")


def load_alpha(folder, paired):
    result = check_result(folder / "result.json", "paired-four-arm" if paired else "full-original")
    paths, patients = read_json(folder / "image_paths.json"), read_json(folder / "patient_ids.json")
    labels = np.load(folder / "labels.npy", allow_pickle=False)
    values, computed = {}, {}
    for arm in ARMS if paired else ["original"]:
        logits = np.load(folder / f"{arm}_logits.npy", allow_pickle=False)
        scores = np.load(folder / f"{arm}_probabilities.npy", allow_pickle=False)
        validate_arrays(paths, patients, labels, logits)
        np.testing.assert_allclose(scores, 1 / (1 + np.exp(-logits.astype(np.float64))), atol=1e-7, rtol=0)
        computed[arm] = metrics(labels, scores)
        assert_metrics(result["branches"][arm], computed[arm], alpha=True)
        values[arm] = logits
    return {"paths": paths, "patients": patients, "labels": labels,
             "logits": values, "metrics": computed, "source": str(folder), "result": result}


def load_rad(folder, paired):
    result = check_result(folder / "result.json", "paired-four-arm" if paired else "full-original")
    rows = read_csv(folder / ("predictions_paired.csv" if paired else "predictions_full.csv"))
    paths, patients = [r["image_path"] for r in rows], [r["patient_id"] for r in rows]
    labels = np.asarray([[int(r["label:" + t]) for t in TASKS] for r in rows], dtype=np.uint8)
    filenames = {"original": "baseline_paired_logits.npy" if paired else "baseline_full_logits.npy",
                 "uniform_eta03": "query_only_paired_logits.npy", "tta_center_original": "original_tta_paired_logits.npy",
                 "tta_center_refined": "refined_paired_logits.npy"}
    native_keys = {"uniform_eta03": "query_only"}
    values, computed = {}, {}
    for arm in ARMS if paired else ["original"]:
        logits = np.load(folder / filenames[arm], allow_pickle=False)
        validate_arrays(paths, patients, labels, logits)
        if arm in ("original", "tta_center_refined"):
            key = "baseline_logit:" if arm == "original" else "refined_logit:"
            csv_scores = np.asarray([[float(r[key + t]) for t in TASKS] for r in rows])
            np.testing.assert_array_equal(logits.astype(np.float64), csv_scores)
        computed[arm] = metrics(labels, logits)
        recorded = result["arms"][native_keys.get(arm, arm)] if paired else result["baseline_full"]
        assert_metrics(recorded, computed[arm], alpha=False)
        values[arm] = logits
    if paired:
        center = np.load(folder / "center_only_paired_logits.npy", allow_pickle=False)
        np.testing.assert_array_equal(values["tta_center_refined"],
                                      (values["uniform_eta03"] + center) * np.float32(.5))
    return {"paths": paths, "patients": patients, "labels": labels,
             "logits": values, "metrics": computed, "source": str(folder), "result": result}


def verify_subset(full, paired):
    lookup = {p: i for i, p in enumerate(full["paths"])}
    ids = [lookup[p] for p in paired["paths"]]
    np.testing.assert_array_equal(full["labels"][ids], paired["labels"])
    if [full["patients"][i] for i in ids] != paired["patients"]:
        raise ValueError("Patient identity differs between full and paired predictions")
    a, b = full["logits"]["original"][ids], paired["logits"]["original"]
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=0,
                               err_msg="Full baseline subset and independently run paired baseline disagree")
    return {"passed": True, "matched_images": len(ids), "max_abs_logit_error": float(np.abs(a - b).max()),
             "logits_bitwise_equal": bool(np.array_equal(a, b)), "atol": 2e-5, "rtol": 0}


def verify_manifest(data, path):
    rows = read_csv(path)
    paths = [r.get("image_path") or r["image"] for r in rows]
    if paths != data["paths"] or [r["patient_id"] for r in rows] != data["patients"]:
        raise ValueError(f"Prediction coverage/order differs from manifest {path}")
    labels = np.asarray([[int(r[t]) for t in TASKS] for r in rows], dtype=np.uint8)
    np.testing.assert_array_equal(labels, data["labels"])


def verify_startup_provenance(folder, *, alpha):
    """Detect post-start source, prompt, or manifest drift before reporting."""
    recorded = {}
    if alpha:
        recorded.update(read_json(folder / "source_sha256.json"))
        prompt = read_json(folder / "prompt_provenance.json")
        recorded[prompt["source"]] = prompt["sha256"]
        check = read_json(folder / "zero_eta_check.json")
    else:
        config = read_json(folder / "run_config.json")
        recorded.update(config["source_sha256"])
        recorded.update(config["manifest_sha256"])
        recorded[config["catalog"]] = config["catalog_sha256"]
        check = read_json(folder / "reference_equivalence.json")
    if check.get("passed") is not True:
        raise ValueError(f"Original published-helper equivalence check failed: {folder}")
    for path, expected in recorded.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Source/input changed since job startup: {path}")
    return {"passed": True, "source_or_input_count": len(recorded),
             "sha256": recorded, "original_helper_check": check}


def summarize(run):
    startup = {name: verify_startup_provenance(run / name, alpha=name.startswith("alpha"))
               for name in ("alpharad_full", "alpharad_paired", "radzero_full", "radzero_paired")}
    alpha_full, alpha_pair = load_alpha(run / "alpharad_full", False), load_alpha(run / "alpharad_paired", True)
    rad_full, rad_pair = load_rad(run / "radzero_full", False), load_rad(run / "radzero_paired", True)
    for a, b in [(alpha_full, rad_full), (alpha_pair, rad_pair)]:
        if a["paths"] != b["paths"] or a["patients"] != b["patients"]:
            raise ValueError("Models used different image or patient order")
        np.testing.assert_array_equal(a["labels"], b["labels"])
    if len(alpha_full["paths"]) != 7494 or len(alpha_pair["paths"]) != 7407:
        raise ValueError("Unexpected full/paired dataset sizes")
    if len(set(alpha_full["patients"])) != 2500 or len(set(alpha_pair["patients"])) != 2470:
        raise ValueError("Unexpected full/paired patient counts")
    catalog_input = read_json(run / "alpharad_paired/catalog_input.json")
    if sha(Path(catalog_input["path"])) != catalog_input["sha256"]:
        raise ValueError("Frozen input catalog changed")
    support_patients, support_paths, provenance = set(), set(), {}
    for task, entry in catalog_input["entries"].items():
        for key, hashkey in [("csv_path", "csv_sha256"), ("original_csv_path", "original_csv_sha256"),
                             ("support_csv_path", "support_csv_sha256"), ("split_audit_path", "split_audit_sha256")]:
            path = Path(entry[key])
            if sha(path) != entry[hashkey]:
                raise ValueError(f"Frozen dataset input changed: {path}")
            provenance[str(path)] = entry[hashkey]
        verify_manifest(alpha_pair, Path(entry["csv_path"]))
        verify_manifest(alpha_full, Path(entry["original_csv_path"]))
        support = read_csv(Path(entry["support_csv_path"]))
        if len(support) != 10 or sum(int(r[task]) for r in support) != 5 or len({r["patient_id"] for r in support}) != 10:
            raise ValueError("Wrong support-set composition")
        support_patients.update(r["patient_id"] for r in support)
        support_paths.update(r.get("image_path") or r["image"] for r in support)
    if len(support_patients) != 30 or len(support_paths) != 30:
        raise ValueError("Unexpected support union size")
    if support_patients & set(alpha_pair["patients"]):
        raise ValueError("Support/test patient leakage")
    removed = set(alpha_full["paths"]) - set(alpha_pair["paths"])
    expected_removed = {p for p, pid in zip(alpha_full["paths"], alpha_full["patients"]) if pid in support_patients}
    if removed != expected_removed or len(removed) != 87:
        raise ValueError("The held-out manifest did not exclude exactly all support patients' images")
    subset = {"AlphaRAD": verify_subset(alpha_full, alpha_pair), "RadZero": verify_subset(rad_full, rad_pair)}
    records, ablation, full_records = [], [], []
    for model, full, pair in [("AlphaRAD", alpha_full, alpha_pair), ("RadZero", rad_full, rad_pair)]:
        for task in TASKS:
            base = pair["metrics"]["original"]["per_task"][task]
            ours = pair["metrics"]["tta_center_refined"]["per_task"][task]
            row = {"model": model, "task": task, "display_name": DISPLAY[task], "test_images": 7407,
                    "test_patients": 2470, "support_images_per_task": 10, "positive_test_images": base["positive"]}
            for key in KEYS:
                row["baseline_" + key] = base[key] * 100
                row["ours_" + key] = ours[key] * 100
                row["delta_" + key + "_pp"] = (ours[key] - base[key]) * 100
            records.append(row)
            full_records.append({"model": model, "task": task,
                                 **full["metrics"]["original"]["per_task"][task]})
            for arm in ARMS:
                ablation.append({"model": model, "task": task, "arm": arm,
                                 **pair["metrics"][arm]["per_task"][task]})
    for folder in ("alpharad_full", "alpharad_paired", "radzero_full", "radzero_paired"):
        for path in sorted((run / folder).iterdir()):
            if path.suffix in (".json", ".npy", ".csv"):
                provenance[str(path)] = sha(path)
    payload = {"status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "run": str(run), "comparison": records, "full_baseline": full_records, "ablations": ablation,
               "baseline_parity": subset, "startup_provenance_checks": startup,
               "all_models_identical_images_labels_and_patient_order": True,
               "metric_unit": "image", "table_units": "percent; differences in percentage points",
               "auprc_definition": "trapezoidal area under precision-recall curve",
               "ap_definition": "average_precision_score, reported separately",
               "full_image_count": 7494, "full_patient_count": 2500,
               "paired_image_count": 7407, "paired_patient_count": 2470,
               "support_union_images": 30, "support_union_patients": 30, "removed_images": 87,
               "provenance_sha256": provenance,
               "limitations": ["Custom subset, not the AlphaRAD paper's exact test split.",
                               "Explicit reconstructed English queries; exact author CheXchoNet prompts are unavailable.",
                               "Labelled support images were selected from the custom test source, then all images of their patients were excluded.",
                               "Image-level metrics; repeated images from a patient are not statistically independent.",
                               "Fixed seed42, 5 positive and 5 negative patients per task, eta=.3, center80; no test-driven hyperparameter search."]}
    save_json(run / "RESULTS.json", payload)
    with (run / "COMPARISON.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["CheXchoNet custom 7494：AlphaRAD / RadZero 与固定查询修正＋双视图平均", "",
             "状态：4个任务进程全部完成；2个模型×3项二分类任务，共6组公平对照。",
             "CSLV：严重左心室肥厚；CDLV：左心室扩张；CCOM：两者任一阳性（逻辑OR）。三项任务使用同一批胸片、不同标签。",
             "全量：7,494张/2,500名患者。每任务支持5阳性＋5阴性，共30张/30名患者；这些患者全部87张图从测试中排除。",
             "公平对照：7,407张/2,470名患者。两模型、全部方法和全部任务使用完全相同的测试图片顺序、患者和标签。",
             "方法固定：η=0.3；冻结模型；原图＋标准预处理后中心80%宽高裁剪；原始logit等权平均。无反向传播训练。",
             "注意：这是自定义CheXchoNet子集，不是论文作者可核实的原测试划分。论文任务含义相同，不代表数据划分相同。",
             "官方仓库未公开CheXchoNet完整提示词；本次明确重建疾病英文描述并使用各模型原生文本编码/图文匹配。",
             "单位：%；差值：百分点（未四舍五入值计算）。AUPRC为PR曲线梯形面积；AP另列，二者不混用。", "",
             "| 模型 | 任务 | 原始AUROC | 本方法AUROC | ΔAUROC | 原始AUPRC | 本方法AUPRC | ΔAUPRC | 原始AP | 本方法AP | ΔAP |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        fields = [f"{r['model']}", DISPLAY[r["task"]]]
        for k in KEYS:
            fields += [f"{r['baseline_'+k]:.2f}", f"{r['ours_'+k]:.2f}", f"{r['delta_'+k+'_pp']:+.2f}"]
        lines.append("| " + " | ".join(fields) + " |")
    lines += ["", "完整7,494张原始模型参考（不与7,407张本方法直接相减）：", "",
              "| 模型 | 任务 | AUROC | AUPRC | AP |", "|---|---|---:|---:|---:|"]
    for r in full_records:
        lines.append(f"| {r['model']} | {DISPLAY[r['task']]} | {r['auroc']*100:.2f} | {r['auprc']*100:.2f} | {r['average_precision']*100:.2f} |")
    lines += ["", "同一7,407张测试集的四臂消融：", "",
              "| 模型 | 任务 | 方法 | AUROC | AUPRC | AP |", "|---|---|---|---:|---:|---:|"]
    for r in ablation:
        lines.append(f"| {r['model']} | {DISPLAY[r['task']]} | {ARM_NAMES[r['arm']]} | {r['auroc']*100:.2f} | {r['auprc']*100:.2f} | {r['average_precision']*100:.2f} |")
    lines += ["", "数值与数据核验："]
    for model, audit in subset.items():
        lines.append(f"- {model}：全量原始预测按图片路径取7,407张子集，与独立配对原始预测一致；最大logit绝对误差{audit['max_abs_logit_error']:.8g}（阈值2e-5，无相对容差）。")
    lines += ["- AUROC、AUPRC、AP均由保存预测独立重算；已有result.json中的指标全部核验一致。",
              "- 四个任务启动时记录的源码、数据清单与提示词SHA256在汇总时全部保持一致；原模型helper一致性核验全部通过。",
              "- AlphaRAD按原生sigmoid分数评估；RadZero沿用原生logit评估；均在每类内部计算，再列出单任务结果。",
              "- 重复患者图像仍按图片计分；本表不是患者级评估，也未给出患者聚类置信区间。",
              "- 保留全部提升/下降结果，不依据测试成绩重新抽支持集或选择η。"]
    (run / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "comparisons": len(records), "run": str(run),
                      "baseline_parity": subset}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run.resolve())


if __name__ == "__main__":
    main()
