#!/usr/bin/env python3
"""Render and audit frozen CheXchoNet same-view comparator publication tables.

Reads completed results only; never fits models or opens test-label assets.
Publication sources are SHA256 pinned and checked against the independent audit.
The manuscript checker parses table cells and indexes raw JSON records directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "experiments/fewshot_neighbors_20261025"
DEFAULT_RUN = CAMPAIGN / "runs/full_20261025_v1"
MODELS = ("BiomedCLIP", "AlphaRAD")
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SCOPES = TASKS + ("macro_three_tasks",)
TASK_LABELS = dict(zip(TASKS, ("CSLV", "CDLV", "CCOM")))
SEEDS = (42, 43, 44, 45, 46)
PRIMARY_METRICS = ("auroc", "auprc")
METRICS = PRIMARY_METRICS + ("average_precision",)
SINGLE_METHODS = ("original", "query", "prototype", "prototype_mix", "tip_adapter", "gda", "simnl", "project_mix", "linear_probe")
METHODS = SINGLE_METHODS + tuple(name + "_view" for name in SINGLE_METHODS)
COMPETITORS = SINGLE_METHODS[2:]
METHOD_LABELS = dict(zip(SINGLE_METHODS, ("Original", "Q", "原型分类器", "原型分数混合", "Tip-Adapter", "GDA", "SimNL", "P&M", "线性探针")))
VIEWS = (("单视图", ""), ("双视图", "_view"))
PINNED_SHA256 = {
    "RESULTS.json": "388dd3138e4075fa0700a2f4e4747401f97747876a00d61079022fd80f0f3e16",
    "INDEPENDENT_AUDIT.json": "108170ea99e666a9e3ab7afda30f5df0e62e3792098f3a5636ea7f412b3a21d3",
    "FROZEN_RUN_PLAN.json": "c1552353141d3e011f1f5ea6c26ec61a6a3b2fa60f976b99f7e8093053fc9694",
    "FIT_COMPLETION_AUDIT.json": "8ec076b3413ced0daf826c720ef742b1fdf610e199bfdceeca146b0ea843511a",
}
PROTOCOL_SHA256 = "b240185791f8396d7a7f4cff13872e5ffa7405c8877f08d490e1123260ac0081"
TABLE_SPECS = (
    ("mean_sd_results", "MEAN_SD_COMPARISON.csv", ("model", "task", "branch", "metric"), 432),
    ("paired_comparisons", "PAIRED_COMPARISONS.csv", ("model", "task", "comparison", "metric"), 600),
    ("per_seed_results", "PER_SEED_RESULTS.csv", ("model", "support_seed", "task", "branch", "metric"), 2160),
    ("per_seed_paired_comparisons", "PER_SEED_PAIRED_COMPARISONS.csv", ("model", "support_seed", "task", "comparison", "metric"), 3000),
)
HEADER15 = ["底座", "任务", "视图", "Original（同视图）", "Q／Q+V", "Project and Mix", "线性探针"]
HEADER16 = ["底座", "任务", "视图", "查询分支−P&M：ΔAUROC", "查询分支−P&M：ΔAUPRC", "查询分支−线性探针：ΔAUROC", "查询分支−线性探针：ΔAUPRC"]
HEADERG = ["底座", "视图", "方法", "CSLV（AUROC／AUPRC）", "CDLV（AUROC／AUPRC）", "CCOM（AUROC／AUPRC）"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Neighbor-comparison publication verification failed: " + message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lookups(summary: dict[str, Any]) -> tuple[dict, dict]:
    means = {(r["model"], r["task"], r["branch"], r["metric"]): r for r in summary["mean_sd_results"]}
    pairs = {(r["model"], r["task"], r["comparison"], r["metric"]): r for r in summary["paired_comparisons"]}
    return means, pairs


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

    for name, digest in PINNED_SHA256.items():
        check_hash(run / name, digest)
    summary = _read_json(run / "RESULTS.json")
    audit_record = _read_json(run / "INDEPENDENT_AUDIT.json")
    plan = _read_json(run / "FROZEN_RUN_PLAN.json")
    fits = _read_json(run / "FIT_COMPLETION_AUDIT.json")
    check_hash(summary["protocol"], PROTOCOL_SHA256)
    protocol = _read_json(Path(summary["protocol"]))
    experiment = protocol["experiment"]
    _require(summary["status"] == "complete" and audit_record["status"] == "passed" and fits["status"] == "verified", "completion/audit status")
    _require(summary["models"] == plan["models"] == fits["complete_models"] == experiment["models"] == list(MODELS), "model scope")
    _require(set(audit_record["models"]) == set(MODELS), "independent model scope")
    _require(summary["task_order"] == plan["tasks"] == experiment["task_order"] == list(TASKS), "task scope")
    _require(summary["support_seeds"] == plan["seeds"] == experiment["support_seeds"] == list(SEEDS), "support seed scope")
    _require(summary["methods"] == experiment["methods"] == list(METHODS), "method order/scope")
    _require(summary["competitors"] == experiment["competitors"] == list(COMPETITORS), "competitor scope")
    _require(summary["test_images"] == 7407 and summary["test_patients"] == 2470, "test scope")
    _require(summary["task_seed_branch_results"] == plan["task_seed_branch_results"] == fits["task_seed_branch_results"] == audit_record["task_seed_branch_results"] == 540, "complete task/seed/branch count")
    _require(summary["protocol_sha256"] == audit_record["protocol_sha256"] == PROTOCOL_SHA256, "protocol identity")
    _require(audit_record["results_sha256"] == PINNED_SHA256["RESULTS.json"], "audited result identity")
    _require(summary["fit_completion_audit_sha256"] == PINNED_SHA256["FIT_COMPLETION_AUDIT.json"], "fit commitment identity")
    _require(summary["run_plan_audit"]["sha256"] == PINNED_SHA256["FROZEN_RUN_PLAN.json"], "frozen run identity")
    _require(summary["metric_names"] == list(METRICS) and summary["primary_metrics"] == list(PRIMARY_METRICS), "metric definitions")
    _require(summary["bootstrap_replicates"] == 2000 and summary["bootstrap_seed"] == 20261024, "paired bootstrap configuration")
    _require(summary["all_results_retained"] and summary["no_cross_model_averaging"] and plan["all_results_retained"], "outcome retention policy")
    _require(audit_record["all_metric_points_independently_recomputed"] and audit_record["all_intervals_recomputed_from_saved_paired_draw_metrics"], "independent numerical audit")
    _require(audit_record["native_predictions_unchanged"] and audit_record["support_folds_and_single_dual_fit_metadata_match"], "same-support/same-fit audit")
    _require(audit_record["test_labels_read_after_all_fits_committed"] and not fits["test_labels_opened"] and not plan["test_labels_opened_for_fitting"], "test-label separation")
    _require(experiment["eta"] == .3 and experiment["center_crop_ratio"] == .8 and experiment["views"] == ["identity", "center80"], "fixed views/query configuration")
    _require(experiment["selection"]["metric"] == "pooled_oof_trapezoidal_auprc" and experiment["selection"]["select_once_for_single_and_dual"], "support-only selection policy")
    _require(summary["study_type"] == "exploratory_same_support_same_view_neighbor_comparison", "study interpretation")
    for source_map in (plan["sources"], summary["source_sha256"], experiment["algorithm_sources_sha256"]):
        for path, digest in source_map.items():
            check_hash(path, digest)

    row_counts = {}
    for key, filename, identity, expected in TABLE_SPECS:
        records = summary[key]
        _require(len(records) == expected and len({tuple(r[k] for k in identity) for r in records}) == expected, key + " row uniqueness/count")
        _require({r["model"] for r in records} == set(MODELS) and {r["task"] for r in records} == set(SCOPES), key + " model/task coverage")
        _require({r["metric"] for r in records} == set(METRICS), key + " metric coverage")
        with (run / filename).open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        _require(len(csv_rows) == expected, filename + " row count")
        for row, csv_row in zip(records, csv_rows):
            _require(set(row) == set(csv_row) and all(csv_row[k] == ("" if v is None else str(v)) for k, v in row.items()), filename + " differs from unrounded JSON")
            _require(all(not isinstance(v, float) or math.isfinite(v) for v in row.values()), filename + " nonfinite value")
        checked[str(run / filename)] = _sha256(run / filename)
        row_counts[key] = expected

    means, pairs = _lookups(summary)
    seed_points = {(r["model"], r["task"], r["support_seed"], r["branch"], r["metric"]): r["metric_percent"] for r in summary["per_seed_results"]}
    for key, row in means.items():
        model, task, branch, metric = key
        points = [seed_points[model, task, seed, branch, metric] for seed in SEEDS]
        _require(math.isclose(statistics.mean(points), row["mean_percent"], abs_tol=1e-10), "mean/seed inconsistency")
        _require(math.isclose(statistics.stdev(points), row["support_seed_sd_pp"], abs_tol=1e-10), "sample SD inconsistency")
    for row in pairs.values():
        point_difference = means[row["model"], row["task"], row["high_branch"], row["metric"]]["mean_percent"] - means[row["model"], row["task"], row["low_branch"], row["metric"]]["mean_percent"]
        _require(math.isclose(point_difference, row["mean_delta_pp"], abs_tol=1e-10), "paired difference inconsistency")
        _require(row["valid_replicates"] == 2000 and row["invalid_replicates"] == 0 and row["confidence_level"] == .95, "confidence interval coverage/definition")
    count = len({(r["model"], r["task"], r["support_seed"], r["branch"]) for r in summary["per_seed_results"] if r["task"] in TASKS})
    _require(count == 540, "complete task/seed/branch identity count")
    record = dict(status="verified", run=str(run), models=list(MODELS), tasks=list(TASKS), support_seeds=list(SEEDS),
                  methods=list(METHODS), primary_metrics=list(PRIMARY_METRICS), task_seed_branch_results=count,
                  task_method_means=108, test_images=7407, test_patients=2470, bootstrap_replicates=2000,
                  independent_audit_status=audit_record["status"], row_counts=row_counts,
                  source_files_sha256=checked, publication_decimals=dict(table15=2, table16=4, appendix_g=2))
    return summary, record


def provenance(run_dir: str | Path | None = None) -> dict[str, Any]:
    return _load_verified(run_dir)[1]


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(row) + " |" for row in rows])


def table15(run_dir: str | Path | None = None) -> str:
    """Twelve same-view rows: Original, Q, Project and Mix, linear probe."""
    summary, _ = _load_verified(run_dir)
    means, _ = _lookups(summary)
    rows = []
    for model in MODELS:
        for task in TASKS:
            for view, suffix in VIEWS:
                row = [model, TASK_LABELS[task], view]
                for method in ("original", "query", "project_mix", "linear_probe"):
                    row.append("／".join(f"{means[model, task, method + suffix, metric]['mean_percent']:.2f}" for metric in PRIMARY_METRICS))
                rows.append(row)
    return _table(HEADER15, rows)


def table16(run_dir: str | Path | None = None) -> str:
    """Twelve rows retaining all task/view Q-minus-P&M and Q-minus-LP intervals."""
    summary, _ = _load_verified(run_dir)
    _, pairs = _lookups(summary)
    rows = []
    for model in MODELS:
        for task in TASKS:
            for view, suffix in VIEWS:
                row = [model, TASK_LABELS[task], view]
                for method in ("project_mix", "linear_probe"):
                    comparison = "query" + suffix + "_minus_" + method + suffix
                    for metric in PRIMARY_METRICS:
                        r = pairs[model, task, comparison, metric]
                        row.append(f"{r['mean_delta_pp']:+.4f}［{r['ci_low_pp']:+.4f}, {r['ci_high_pp']:+.4f}］")
                rows.append(row)
    return _table(HEADER16, rows)


def _appendix_table(summary: dict[str, Any], model: str) -> str:
    means, _ = _lookups(summary)
    rows = []
    for view, suffix in VIEWS:
        for method in SINGLE_METHODS:
            label = "Q+V" if method == "query" and suffix else METHOD_LABELS[method]
            row = [model, view, label]
            for task in TASKS:
                cells = []
                for metric in PRIMARY_METRICS:
                    r = means[model, task, method + suffix, metric]
                    cells.append(f"{r['mean_percent']:.2f} ± {r['support_seed_sd_pp']:.2f}")
                row.append("／".join(cells))
            rows.append(row)
    return _table(HEADERG, rows)


def appendix_g(run_dir: str | Path | None = None) -> str:
    """Complete 108 task/method means with sample SD in two compact tables."""
    summary, _ = _load_verified(run_dir)
    source = "../../experiments/fewshot_neighbors_20261025/runs/full_20261025_v1/"
    text = [
        "## 附录 G CheXchoNet 同支持、同视图的近邻方法完整比较",
        "本附录保留 BiomedCLIP、AlphaRAD 两个底座、CSLV／CDLV／CCOM 三个任务的全部 18 个方法／视图条件，"
        "共 108 个任务×方法均值，对应五支持种子（42–46）的 **540 项任务×种子×分支结果**。"
        "固定测试为 7,407 张图、2,470 位患者，三个标签共用测试图；每任务、每种子使用五名阳性与五名严格阴性支持患者的各一张图，"
        "支持与测试患者无交集。不同种子的支持集可能重叠。此表不增加既有 135 项主评估任务数。",
        "每格依次为 **AUROC 均值 ± 样本标准差／AUPRC 均值 ± 样本标准差**；指标为百分数，标准差为百分点，"
        "AUPRC 使用 PR 曲线梯形面积。先计算各支持种子的指标，再计算五种子均值与样本标准差，不集成五组预测。"
        "单视图为原图，双视图为原图与边长比例 0.8 的中心裁剪放大图。双视图各方法均沿用原图支持上选定的同一拟合结果，输出等权融合。"
        "Original 的双视图行为原有 V 分支，不使用支持标签；Q 的双视图行为 Q+V。",
        "七种竞争方法均只使用本任务的十张带标签支持图；需要选参的方法采用共享的五折患者分组验证，以合并折外预测的梯形 AUPRC 选参。"
        "Q 固定 η＝0.3。Project and Mix（P&M）依据论文公式重实现为每任务正／负两类文本的移植版，"
        "本轮不使用原论文的十次支持增强，因此不等同于作者原始协议的精确复现；线性探针为带截距的 L2 逻辑回归。"
        "所有骨干冻结，线性探针与 SimNL 的小型参数仍需拟合。",
        "BiomedCLIP 的查询和竞争方法使用共享全局投影空间；AlphaRAD 的 Q 使用查询相关的类局部支持统计，"
        "竞争分数方法使用冻结全局投影视觉特征。因此相同标签和视图预算不等于相同中间表征。"
        "双视图中 AlphaRAD 原生分支先平均 logit 再取 sigmoid，BiomedCLIP 原生分支平均概率，竞争方法平均各自 margin／logit。",
        "### G.1 BiomedCLIP：全部方法和视图",
        _appendix_table(summary, "BiomedCLIP"),
        "表 G1：BiomedCLIP 三任务的单／双视图完整均值与标准差。所有方法和支持种子均保留，不以测试成绩切换主方法。",
        "### G.2 AlphaRAD：全部方法和视图",
        _appendix_table(summary, "AlphaRAD"),
        "表 G2：AlphaRAD 三任务的单／双视图完整均值与标准差。Q 与全局特征竞争方法的中间表征差异如上所述。",
        "表 16 列出 查询分支相对 P&M、线性探针的全部同视图配对区间；全部七对手比较、查询分支相对 Original 以及双视图相对单视图的全部预设差值，"
        "包括负向、混合方向和区间跨零的结果，均保存在下列完整文件。95% 区间来自 2,000 次共享患者级配对 bootstrap，"
        "条件于当前五组固定支持，不包含未来支持抽样的总体不确定性，未作多重比较校正。"
        "三任务宏平均先在每个种子／抽样内平均任务指标，不增加独立队列数，也不作跨底座合并平均。",
        "CheXchoNet 此前已用于整体适配和查询路径评估；本轮预先固定比较分支、支持折和候选范围，并在全部拟合与预测保存后评分，"
        "但仍属于已有数据上的新比较，不能改称从未使用过的独立队列。局部比较优势不意味着所有任务相对 Original 均提高，也不意味着普遍优于全部对手。",
        f"完整数据见[全部结果]({source}RESULTS.json)、[108 个任务×方法均值及宏平均]({source}MEAN_SD_COMPARISON.csv)、"
        f"[全部预设配对区间]({source}PAIRED_COMPARISONS.csv)、[逐种子结果]({source}PER_SEED_RESULTS.csv)、"
        f"[逐种子配对结果]({source}PER_SEED_PAIRED_COMPARISONS.csv)及[独立统计审计]({source}INDEPENDENT_AUDIT.json)。"
        "生成表格时核验结果、协议、冻结运行计划和审计哈希，并逐项检查 CSV 与未舍入 JSON 一致。",
    ]
    return "\n\n".join(text) + "\n"


def _markdown_tables(markdown: str) -> list[list[list[str]]]:
    tables, block = [], []
    for line in markdown.splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            block.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif block:
            if len(block) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in block[1]):
                tables.append([block[0]] + block[2:])
            block = []
    return tables


def verify_rendered_tables(markdown: str, run_dir: str | Path | None = None) -> dict[str, Any]:
    """Audit cell identities/numbers from raw JSON; does not compare renderer strings."""
    summary, record = _load_verified(run_dir)
    means, pairs = _lookups(summary)
    tables = _markdown_tables(markdown)
    selected = [[t for t in tables if t[0] == h] for h in (HEADER15, HEADER16, HEADERG)]
    _require([len(x) for x in selected] == [1, 1, 2], "expected one Table 15, one Table 16, and two Appendix G tables")
    task_keys = {v: k for k, v in TASK_LABELS.items()}
    suffixes = dict(VIEWS)
    method_keys = {v: k for k, v in METHOD_LABELS.items()}
    method_keys["Q+V"] = "query"
    checked_numbers = 0

    def check_numbers(cell: str, expected: list[str], context: str) -> None:
        nonlocal checked_numbers
        found = re.findall(r"[+−-]?\d+\.\d+", cell)
        _require(found == expected, context + ": " + cell + " differs from " + str(expected))
        checked_numbers += len(expected)

    rows = selected[0][0][1:]
    expected_keys = {(m, t, v) for m in MODELS for t in TASKS for v, _ in VIEWS}
    found_keys = []
    for row in rows:
        _require(len(row) == len(HEADER15), "Table 15 column count")
        model, task, view = row[0], task_keys.get(row[1]), row[2]
        _require((model, task, view) in expected_keys, "Table 15 row identity")
        found_keys.append((model, task, view))
        for cell, method in zip(row[3:], ("original", "query", "project_mix", "linear_probe")):
            check_numbers(cell, [f"{means[model, task, method + suffixes[view], metric]['mean_percent']:.2f}" for metric in PRIMARY_METRICS], "Table 15")
    _require(len(found_keys) == 12 and set(found_keys) == expected_keys, "Table 15 row completeness")

    found_keys = []
    for row in selected[1][0][1:]:
        _require(len(row) == len(HEADER16), "Table 16 column count")
        model, task, view = row[0], task_keys.get(row[1]), row[2]
        _require((model, task, view) in expected_keys, "Table 16 row identity")
        found_keys.append((model, task, view))
        for cell, (method, metric) in zip(row[3:], [(c, m) for c in ("project_mix", "linear_probe") for m in PRIMARY_METRICS]):
            suffix = suffixes[view]
            comparison = "query" + suffix + "_minus_" + method + suffix
            r = pairs[model, task, comparison, metric]
            check_numbers(cell, [f"{r[field]:+.4f}" for field in ("mean_delta_pp", "ci_low_pp", "ci_high_pp")], "Table 16")
    _require(len(found_keys) == 12 and set(found_keys) == expected_keys, "Table 16 row completeness")

    found_keys = []
    expected_keys = {(m, v, c) for m in MODELS for v, _ in VIEWS for c in SINGLE_METHODS}
    for table in selected[2]:
        _require(len(table[1:]) == 18 and len({r[0] for r in table[1:]}) == 1, "Appendix G per-model row count")
        for row in table[1:]:
            _require(len(row) == len(HEADERG), "Appendix G column count")
            model, view, method = row[0], row[1], method_keys.get(row[2])
            _require((model, view, method) in expected_keys, "Appendix G row identity")
            _require(row[2] != "Q+V" or view == "双视图", "Q+V view label")
            _require(not (method == "query" and view == "双视图") or row[2] == "Q+V", "dual query label")
            found_keys.append((model, view, method))
            for cell, task in zip(row[3:], TASKS):
                expected = []
                for metric in PRIMARY_METRICS:
                    r = means[model, task, method + suffixes[view], metric]
                    expected.extend(f"{r[field]:.2f}" for field in ("mean_percent", "support_seed_sd_pp"))
                check_numbers(cell, expected, "Appendix G")
    _require(len(found_keys) == 36 and set(found_keys) == expected_keys, "Appendix G row completeness")
    _require(checked_numbers == 672, "total numerical-cell token coverage")
    return dict(status="verified", tables=["15", "16", "G1", "G2"], numeric_values_checked=checked_numbers,
                task_seed_branch_results=540, task_method_means=108,
                source_files_sha256=record["source_files_sha256"])


def audit(paths: list[str | Path], run_dir: str | Path | None = None) -> dict[str, Any]:
    files = {}
    for path in paths:
        path = Path(path).resolve()
        result = verify_rendered_tables(path.read_text(encoding="utf-8"), run_dir)
        files[str(path)] = dict(result, manuscript_sha256=_sha256(path))
    return dict(status="verified", files=files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("verify", "table15", "table16", "appendix", "provenance", "audit"), default="verify")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.section in ("table15", "table16", "appendix"):
        function = {"table15": table15, "table16": table16, "appendix": appendix_g}[args.section]
        print(function(args.run), end="\n")
    elif args.section == "audit":
        parser.error("--paths is required with --section audit") if not args.paths else None
        print(json.dumps(audit(args.paths, args.run), ensure_ascii=False, indent=2))
    else:
        record = provenance(args.run)
        print(json.dumps(record if args.section == "provenance" else {k: record[k] for k in ("status", "models", "task_seed_branch_results", "task_method_means", "independent_audit_status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
