#!/usr/bin/env python3
"""Render the audited CheXchoNet query-path extension without running models.

The publication snapshot is pinned by SHA256. CSV tables are checked against
the unrounded JSON records, the frozen model/control scope, and the independent
statistics audit before rendering. All displayed numbers are already in percent
or percentage points; this module never rescales or recomputes model metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "experiments/query_path_multibackbone_20261025"
DEFAULT_RUN = CAMPAIGN / "runs/full_20261025_v1"
MODELS = ("AlphaRAD", "RadZero", "CARZero", "MedKLIP", "MAVL", "KAD")
STRICT_MODELS = MODELS[:2]
DECODER_MODELS = MODELS[2:]
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SCOPES = TASKS + ("macro_three_tasks",)
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_METRICS = ("auroc", "auprc")
METRICS = PRIMARY_METRICS + ("average_precision",)
TASK_LABELS = dict(zip(SCOPES, ("CSLV", "CDLV", "CCOM", "三任务宏平均")))
PINNED_SHA256 = {
    "RESULTS.json": "ebb57d873a0a486117d400cab43dc24bb9e5b0114e58397566c6469f908fd38d",
    "FROZEN_RUN_PLAN.json": "2772f79d0ae2ae31154ad7f3ae28214bd98454dcf83beb65228a0b15ce8b8bd4",
    "INDEPENDENT_STATISTICS_AUDIT.json": "c5c7511f767d530dc14a723134b8a34ac8e5ea065456071d67dde56fd392159e",
}
PROTOCOL_SHA256 = "891abea28f781c36c663c4b83ba6d5d61085d7f233d88607c97192b55ff40a57"
TABLE_SPECS = (
    ("mean_sd_results", "MEAN_SD_COMPARISON.csv", ("model", "task", "branch", "metric"), 216),
    ("paired_comparisons", "PAIRED_COMPARISONS.csv", ("model", "task", "comparison", "metric"), 216),
    ("per_seed_results", "PER_SEED_RESULTS.csv", ("model", "support_seed", "task", "branch", "metric"), 1080),
    ("per_seed_paired_comparisons", "PER_SEED_PAIRED_COMPARISONS.csv",
     ("model", "support_seed", "task", "comparison", "metric"), 1080),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Query-path publication verification failed: " + message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _control(model: str) -> str:
    return "readout_only" if model in STRICT_MODELS else "fixed_aggregation"


def _load_verified(run_dir: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(run_dir or DEFAULT_RUN).resolve()
    checked: dict[str, str] = {}

    def check_hash(path: str | Path, expected: str) -> None:
        path = Path(path).resolve()
        actual = checked.get(str(path))
        if actual is None:
            actual = _sha256(path)
            checked[str(path)] = actual
        _require(actual == expected, "source SHA256 differs: " + str(path))

    for filename, expected in PINNED_SHA256.items():
        check_hash(run / filename, expected)
    summary = _read_json(run / "RESULTS.json")
    plan = _read_json(run / "FROZEN_RUN_PLAN.json")
    audit = _read_json(run / "INDEPENDENT_STATISTICS_AUDIT.json")
    protocol_path = Path(summary["protocol"]).resolve()
    check_hash(protocol_path, PROTOCOL_SHA256)
    protocol = _read_json(protocol_path)
    _require(summary["status"] == "complete" and audit["status"] == "passed", "completion/audit status")
    _require(summary["models"] == plan["models"] == audit["models"] == list(MODELS), "six-model scope/order")
    _require(summary["task_order"] == plan["tasks"] == audit["tasks"] == list(TASKS), "task scope/order")
    _require(summary["support_seeds"] == plan["support_seeds"] == audit["support_seeds"] == list(SEEDS), "support seeds")
    _require(summary["test_images"] == audit["test_images"] == 7407, "test image count")
    _require(summary["test_patients"] == audit["test_patients"] == 2470, "test patient count")
    _require(audit["support_test_patient_overlap"] == 0, "support/test patient isolation")
    _require(plan["eta"] == .3 and plan["views"] == protocol["experiment"]["views"] == ["identity"], "single-view configuration")
    _require(plan["all_models_tasks_seeds_retained"] is True and audit["all_negative_and_positive_results_retained"] is True,
             "full outcome retention")
    _require(summary["study_type"] == "exploratory_multibackbone_mechanism_extension", "exploratory study scope")
    _require(summary["no_cross_model_averaging"] is True and audit["no_cross_model_averaging"] is True, "model averaging policy")
    _require(summary["metric_names"] == list(METRICS) and summary["primary_metrics"] == list(PRIMARY_METRICS), "metric definitions")
    _require(summary["bootstrap_replicates"] == audit["bootstrap_replicates"] == 2000, "bootstrap count")
    _require(summary["bootstrap_seed"] == audit["bootstrap_seed"] == 20261024, "bootstrap seed")
    _require(summary["protocol_sha256"] == plan["protocol_sha256"] == PROTOCOL_SHA256, "frozen protocol identity")
    _require(summary["run_plan_sha256"] == PINNED_SHA256["FROZEN_RUN_PLAN.json"], "frozen run plan identity")
    _require(Path(plan["protocol"]).resolve() == protocol_path, "run plan protocol path")
    _require(audit["task_seed_branch_results"] == 270 and audit["counts"]["archives"] == 30, "complete task/seed/branch count")
    check_hash(CAMPAIGN / "run_campaign.py", plan["controller_source_sha256"])

    for model in MODELS:
        control = _control(model)
        definition = summary["model_controls"][model]
        _require(definition["control_name"] == control and definition["contrast_type"] == "query_minus_" + control,
                 "control definition for " + model)
        expected = "Q - R (readout only)" if model in STRICT_MODELS else "Q - F (fixed visual attention outputs)"
        _require(plan["primary_contrast"][model] == expected, "run plan contrast for " + model)
        _require(audit["model_audits"][model]["primary_contrast"] == definition["contrast_type"], "independent contrast audit")
        worker = summary["worker_metadata_audits"][model]
        _require(worker["status"] == "verified" and worker["control_name"] == control, "worker metadata audit")
        for path, digest in worker["completion_and_metadata_sha256"].items():
            check_hash(path, digest)
        # Worker source manifests are themselves hashed above. Follow their
        # recorded sources so manuscript regeneration detects later code edits.
        for path, digest in _read_json(run / model / "source_sha256.json").items():
            check_hash(path, digest)
    for path, digest in summary["source_sha256"].items():
        check_hash(path, digest)
    for path, digest in audit["source_files_sha256"].items():
        check_hash(path, digest)

    row_counts = {}
    for key, filename, identity, total in TABLE_SPECS:
        records = summary[key]
        _require(len(records) == total, key + " row count")
        _require(len({tuple(row[k] for k in identity) for row in records}) == total, key + " duplicate rows")
        _require({row["model"] for row in records} == set(MODELS), key + " model coverage")
        _require({row["task"] for row in records} == set(SCOPES), key + " task coverage")
        _require({row["metric"] for row in records} == set(METRICS), key + " metric coverage")
        with (run / filename).open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        _require(len(csv_rows) == total, filename + " row count")
        for row, csv_row in zip(records, csv_rows):
            _require(set(row) == set(csv_row), filename + " columns")
            _require(all(csv_row[k] == ("" if v is None else str(v)) for k, v in row.items()), filename + " differs from JSON")
            _require(row["control_name"] == _control(row["model"]), filename + " mixed R/F control")
            _require(row["contrast_type"] == "query_minus_" + _control(row["model"]), filename + " mixed R/F contrast")
            for value in row.values():
                if isinstance(value, float):
                    _require(math.isfinite(value), filename + " nonfinite value")
        checked[str(run / filename)] = _sha256(run / filename)
        row_counts[key] = total

    task_seed_branch_count = len({(r["model"], r["support_seed"], r["task"], r["branch"])
                                 for r in summary["per_seed_results"] if r["task"] in TASKS})
    _require(task_seed_branch_count == 270, "retained task/seed/branch results")
    provenance_record = dict(
        status="verified", run=str(run), study_type=summary["study_type"],
        models=list(MODELS), tasks=list(TASKS), support_seeds=list(SEEDS),
        test_images=7407, test_patients=2470, support_test_patient_overlap=0,
        views=["identity"], eta=.3, bootstrap_replicates=2000, bootstrap_seed=20261024,
        model_controls=summary["model_controls"], primary_metrics=list(PRIMARY_METRICS),
        task_seed_branch_results=270, mean_branch_scopes=72, appendix_model_task_scopes=24,
        row_counts=row_counts, decimals=4, independent_audit_status=audit["status"],
        independent_audit_max_abs_errors=audit["max_abs_errors"],
        source_files_sha256=checked,
        publication_case=dict(model="MedKLIP", task="dlv", comparison="query_minus_fixed_aggregation",
                              selected_after_observing_all_results=True),
    )
    return summary, provenance_record


def provenance(run_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify the frozen batch and return JSON-serializable source provenance."""
    return _load_verified(run_dir)[1]


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def _lookups(summary: dict[str, Any]) -> tuple[dict, dict]:
    means = {(r["model"], r["task"], r["branch"], r["metric"]): r for r in summary["mean_sd_results"]}
    pairs = {(r["model"], r["task"], r["comparison"], r["metric"]): r for r in summary["paired_comparisons"]}
    return means, pairs


def _mean_cell(row: dict[str, Any]) -> str:
    return f"{row['mean_percent']:.4f} ± {row['support_seed_sd_pp']:.4f}"


def _difference_cell(row: dict[str, Any]) -> str:
    _require(row["valid_replicates"] == 2000 and row["invalid_replicates"] == 0, "displayed confidence interval coverage")
    _require(row["confidence_level"] == .95 and row["ci_low_pp"] is not None and row["ci_high_pp"] is not None,
             "displayed confidence interval definition")
    return f"{row['mean_delta_pp']:+.4f}［{row['ci_low_pp']:+.4f}, {row['ci_high_pp']:+.4f}］"


def _case_numeric_table(summary: dict[str, Any]) -> str:
    means, pairs = _lookups(summary)
    rows = []
    for branch, label in (("original", "Original"), ("fixed_aggregation", "固定视觉更新 F"), ("query", "完整查询修正 Q")):
        rows.append([label] + [_mean_cell(means["MedKLIP", "dlv", branch, metric]) for metric in PRIMARY_METRICS])
    rows.append(["Q−F"] + [_difference_cell(pairs["MedKLIP", "dlv", "query_minus_fixed_aggregation", metric])
                           for metric in PRIMARY_METRICS])
    return _table(["方法／配对比较", "AUROC", "AUPRC"], rows)


def render_case_table(run_dir: str | Path | None = None) -> str:
    """Return only the Table 14 grid; the manuscript owns its caption and text."""
    summary, _ = _load_verified(run_dir)
    return _case_numeric_table(summary)


def _appendix_numeric_table(summary: dict[str, Any], models: tuple[str, ...], symbol: str) -> str:
    means, pairs = _lookups(summary)
    rows = []
    for model in models:
        control = _control(model)
        for task in SCOPES:
            row = [model, TASK_LABELS[task]]
            for branch in ("original", control, "query"):
                row.append("／".join(_mean_cell(means[model, task, branch, metric]) for metric in PRIMARY_METRICS))
            row.extend(_difference_cell(pairs[model, task, "query_minus_" + control, metric]) for metric in PRIMARY_METRICS)
            rows.append(row)
    return _table(["模型", "任务", "Original（AUROC／AUPRC）", symbol + "（AUROC／AUPRC）", "Q（AUROC／AUPRC）",
                   "Q−" + symbol + "：ΔAUROC［95% CI］", "Q−" + symbol + "：ΔAUPRC［95% CI］"], rows)


def render_appendix(run_dir: str | Path | None = None) -> str:
    """Return the complete Appendix F with all six backbones and both controls."""
    summary, _ = _load_verified(run_dir)
    source = "../../experiments/query_path_multibackbone_20261025/runs/full_20261025_v1/"
    paragraphs = [
        "## 附录 F CheXchoNet 六底座查询路径扩展的完整结果",
        "本轮复用已冻结的 CheXchoNet 五组支持（种子 42–46）及同一批 **7,407 张测试图、2,470 位测试患者**。"
        "每个任务、每个种子使用来自五名阳性与五名严格阴性患者的各一张支持图；支持／测试患者无交集。"
        "全部骨干冻结，采用原图单视图和固定 η＝0.3。三个标签共用测试图，三任务宏平均先在每个支持种子内平均任务指标，"
        "再计算五种子均值与标准差；不平均预测构造集成，也不作跨模型总平均。",
        "AlphaRAD、RadZero 比较 Original、仅改读出 R 与完整查询修正 Q，主要差值为 Q−R。"
        "CARZero、MedKLIP、MAVL、KAD 使用固定视觉更新对照 F：将每层相关视觉 cross-attention 输出钳制为原始查询前向的对应输出，"
        "保留修正查询的残差、自注意力、FFN 及原生分类头计算。CARZero 仅钳制 i2t 路径，其 t2i 路径与 Q 相同。"
        "Q−F 衡量相对于原始查询视觉更新轨迹重新计算的额外作用，包含后续查询状态的交互；这四个底座不具有等价的独立查询读出端口，"
        "因此 Q−F 与 Q−R 分开报告。",
        "各分支单元格依次为 **AUROC／AUPRC**，均表示五种子均值 ± 样本标准差（ddof＝1）；指标为百分数，标准差和差值为百分点。"
        "AUPRC 使用 PR 曲线梯形面积。主要差值的 95% 区间采用 2,000 次患者级配对 bootstrap，种子为 20261024，"
        "同一患者的全部图像随患者成组重采样，所有模型、任务、支持种子和分支共享抽样。"
        "区间条件于这五组固定支持，不包含未来支持抽样的总体不确定性，未作多重比较校正。"
        "所有显示数值保留四位小数，区间是否跨零按未舍入值判断。",
        "正文重点展示的 MedKLIP／CDLV 是查看本轮六底座完整结果后选取的局部案例。"
        "本附录完整保留正向、负向及区间跨零的结果；该展示属于探索性选择，不能据此将被选案例改称预设的独立验证。"
        "CheXchoNet 已用于既有 Original、Q、V、Q+V 扩展评估；本轮是在该数据上的新路径比较，预训练患者重叠尚未完成审计。",
        "### F.1 AlphaRAD 与 RadZero：Q−R",
        _appendix_numeric_table(summary, STRICT_MODELS, "R"),
        "表 F1：两个具有独立查询读出接口的底座。R 与 Q 使用相同修正读出查询，R 的视觉聚合使用原始查询。"
        "三任务宏平均是同三个任务的汇总，不作为额外独立测试队列。",
        "### F.2 四个冻结解码器底座：Q−F",
        _appendix_numeric_table(summary, DECODER_MODELS, "F"),
        "表 F2：四个冻结解码器底座的固定视觉更新对照。F 的干预位置依原生接口定义，CARZero 的 t2i 路径保持与 Q 相同；"
        "这些差值按 Q−F 解释，不能替换为 Q−R。",
        "本轮完整保存六个模型×三个任务×五个支持种子×三个分支，共 **270 项任务×种子×分支结果**；"
        "表 F1、F2 汇总全部分支均值与主要路径差值，原始逐种子结果及 Q−Original、R／F−Original 的差值另存于结果文件。"
        "本轮单视图、五支持种子结果与既有单次支持的 Q+V 主表分别报告，不增加原有 135 项主表配对任务数。",
        f"数据来源为[完整结果]({source}RESULTS.json)、[分支均值与标准差]({source}MEAN_SD_COMPARISON.csv)、"
        f"[全部配对差值]({source}PAIRED_COMPARISONS.csv)、[逐种子指标]({source}PER_SEED_RESULTS.csv)及"
        f"[独立统计审计]({source}INDEPENDENT_STATISTICS_AUDIT.json)。生成表格时核验原始结果、冻结运行计划、协议和审计记录的 SHA256，"
        "并检查 CSV 与未舍入 JSON 一致。",
    ]
    return "\n\n".join(paragraphs) + "\n"


def verify_rendered_tables(markdown: str, run_dir: str | Path | None = None) -> dict[str, Any]:
    """Require all three generated numeric tables in a manuscript string."""
    summary, record = _load_verified(run_dir)
    tables = {"14": _case_numeric_table(summary),
              "F1": _appendix_numeric_table(summary, STRICT_MODELS, "R"),
              "F2": _appendix_numeric_table(summary, DECODER_MODELS, "F")}
    for name, table in tables.items():
        _require(markdown.count(table) == 1, "table " + name + " missing, duplicated or numerically changed")
    return dict(status="verified", tables=list(tables), decimals=4,
                task_seed_branch_results=270, appendix_model_task_scopes=24,
                source_files_sha256=record["source_files_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("verify", "case", "appendix", "provenance"), default="verify")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.section == "case":
        print(render_case_table(args.run))
    elif args.section == "appendix":
        print(render_appendix(args.run), end="")
    elif args.section == "provenance":
        print(json.dumps(provenance(args.run), ensure_ascii=False, indent=2))
    else:
        record = provenance(args.run)
        print(json.dumps({key: record[key] for key in ("status", "models", "task_seed_branch_results",
                                                      "appendix_model_task_scopes", "independent_audit_status")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
