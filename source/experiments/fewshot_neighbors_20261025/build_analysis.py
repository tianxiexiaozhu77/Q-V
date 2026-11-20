#!/usr/bin/env python3
"""Build a Chinese report only after the frozen neighbor campaign is complete.

This post-run utility reads saved aggregate statistics and fitting records. It
does not open test labels, refit methods, select results or modify frozen code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


MODELS = ("BiomedCLIP", "AlphaRAD")
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
SCOPES = TASKS + ("macro_three_tasks",)
SEEDS = tuple(range(42, 47))
METRICS = ("auroc", "auprc")
COMPETITORS = ("prototype", "prototype_mix", "tip_adapter", "gda", "simnl", "project_mix", "linear_probe")
NAMES = dict(original="Original", query="Q", prototype="原型分类器", prototype_mix="原型分数混合",
             tip_adapter="Tip-Adapter", gda="GDA", simnl="SimNL", project_mix="P&M", linear_probe="线性探针")
TASK_NAMES = dict(slvh="CSLV", dlv="CDLV", composite_slvh_dlv="CCOM", macro_three_tasks="三任务宏平均")
EXPECTED_ROWS = dict(mean_sd_results=432, paired_comparisons=600,
                     per_seed_results=2160, per_seed_paired_comparisons=3000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_file(path, expected):
    require(sha(path) == expected, "来源文件哈希不一致：" + str(path))


def complete_inputs(run):
    """Check completion before reading any result or per-fit statistic."""
    status_path = run / "status.json"
    require(status_path.is_file(), "运行状态不存在；不能读取未完成实验结果")
    campaign = read_json(status_path)
    require(campaign.get("status") == "complete", "完整运行尚未完成；暂不读取结果或生成分析")
    result_path = run / "RESULTS.json"
    require(result_path.is_file(), "完整结果文件尚未生成")
    summary = read_json(result_path)
    require(summary.get("status") == "complete", "统计结果尚未完成")
    require(summary.get("models") == list(MODELS) and summary.get("task_order") == list(TASKS)
            and summary.get("support_seeds") == list(SEEDS), "模型／任务／种子范围不完整")
    require(summary.get("test_images") == 7407 and summary.get("test_patients") == 2470
            and summary.get("task_seed_branch_results") == 540
            and summary.get("all_results_retained") is True, "样本或分支范围不完整")
    for key, count in EXPECTED_ROWS.items():
        require(len(summary[key]) == count, "统计行数不完整：" + key)
    for source, digest in summary["source_sha256"].items():
        check_file(source, digest)
    fit_audit_path = run / "FIT_COMPLETION_AUDIT.json"
    check_file(fit_audit_path, summary["fit_completion_audit_sha256"])
    fit_audit = read_json(fit_audit_path)
    require(fit_audit.get("status") == "verified" and fit_audit.get("total_task_seed_fits") == 30,
            "拟合完成审计不完整")
    for source, digest in summary["run_plan_audit"]["sources"].items():
        check_file(source, digest)
    fits, caches = {}, {}
    for model in MODELS:
        fits[model] = {}
        cache_path = run / model / "cache/result.json"
        cache = read_json(cache_path)
        require(cache["status"] == "complete", "缓存尚未完成：" + model)
        caches[model] = cache
        for seed in SEEDS:
            folder = run / model / "evaluation" / f"seed_{seed}"
            commit_path = folder / "fit_commit.json"
            expected = fit_audit["models"][model]["seed_commits"][str(seed)]
            check_file(commit_path, expected["fit_commit_sha256"])
            commit = read_json(commit_path)
            check_file(cache_path, commit["cache_result_sha256"])
            check_file(folder / "fit_info.json", commit["fit_info_sha256"])
            check_file(folder / "predictions.npz", commit["predictions_sha256"])
            fits[model][seed] = read_json(folder / "fit_info.json")
            require(set(fits[model][seed]) == set(TASKS), "支持拟合任务不完整")
    audit_path = run / "INDEPENDENT_AUDIT.json"
    independent = read_json(audit_path) if audit_path.exists() else {"status": "pending"}
    require(independent.get("status") != "failed", "独立统计审计失败；先检查审计记录")
    metadata = dict(run=str(run), results_sha256=sha(result_path), fit_audit_sha256=sha(fit_audit_path),
                    independent_audit_sha256=sha(audit_path) if audit_path.exists() else None,
                    generator_sha256=sha(__file__), generated_at_utc=datetime.now(timezone.utc).isoformat())
    return summary, fits, caches, independent, metadata


def indexed_rows(summary):
    means = {(r["model"], r["task"], r["branch"], r["metric"]): r for r in summary["mean_sd_results"]}
    pairs = {(r["model"], r["task"], r["high_branch"], r["low_branch"], r["metric"]): r
             for r in summary["paired_comparisons"]}
    seeds = {(r["model"], r["task"], r["high_branch"], r["low_branch"], r["metric"], r["support_seed"]): r
             for r in summary["per_seed_paired_comparisons"]}
    require(len(means) == len(summary["mean_sd_results"]) and len(pairs) == len(summary["paired_comparisons"])
            and len(seeds) == len(summary["per_seed_paired_comparisons"]), "存在重复统计键")
    return means, pairs, seeds


def numeric(value, digits=4, signed=False):
    if value is None:
        return "不可用"
    require(math.isfinite(value), "非有限报告数值")
    result = format(value, ("+" if signed else "") + f".{digits}f")
    # Preserve a sign that fixed-point rounding would hide near zero.
    if signed and value != 0 and float(result) == 0:
        return format(value, ("+" if signed else "") + ".3e")
    return result


def mean_cell(row):
    return numeric(row["mean_percent"]) + " ± " + numeric(row["support_seed_sd_pp"])


def pair_cell(row):
    lo, hi = row["ci_low_pp"], row["ci_high_pp"]
    interval = "区间不可用" if lo is None or hi is None else "［" + numeric(lo, signed=True) + ", " + numeric(hi, signed=True) + "］"
    positive, negative = row["positive_gain_seed_count"], row["negative_gain_seed_count"]
    return (numeric(row["mean_delta_pp"], signed=True) + " " + interval
            + f"；种子 +/−/＝ {positive}/{negative}/{len(SEEDS)-positive-negative}")


def both_counts(pairs, model, tasks, high, low):
    counts = dict(both_positive=0, both_negative=0, mixed=0, zero_or_tie=0,
                  both_ci_positive=0, both_ci_negative=0, other_ci=0)
    for task in tasks:
        rows = [pairs[model, task, high, low, metric] for metric in METRICS]
        values = [r["mean_delta_pp"] for r in rows]
        if all(v > 0 for v in values): counts["both_positive"] += 1
        elif all(v < 0 for v in values): counts["both_negative"] += 1
        elif values[0] * values[1] < 0: counts["mixed"] += 1
        else: counts["zero_or_tie"] += 1
        if all(r["ci_low_pp"] is not None and r["ci_low_pp"] > 0 for r in rows): counts["both_ci_positive"] += 1
        elif all(r["ci_high_pp"] is not None and r["ci_high_pp"] < 0 for r in rows): counts["both_ci_negative"] += 1
        else: counts["other_ci"] += 1
    return counts


def joint_seed_wins(seed_pairs, model, task, high, low):
    return sum(all(seed_pairs[model, task, high, low, metric, seed]["delta_pp"] > 0 for metric in METRICS)
               for seed in SEEDS)


def interpretation(pairs):
    alpha_pm = [both_counts(pairs, "AlphaRAD", TASKS, "query" + suffix, "project_mix" + suffix)
                for suffix in ("", "_view")]
    alpha_lp = [both_counts(pairs, "AlphaRAD", TASKS, "query" + suffix, "linear_probe" + suffix)
                for suffix in ("", "_view")]
    zero_tasks = [TASK_NAMES[task] for task in TASKS
                  if pairs["AlphaRAD", task, "query", "original", "auprc"]["ci_low_pp"] <= 0
                  <= pairs["AlphaRAD", task, "query", "original", "auprc"]["ci_high_pp"]]

    def contrast(model, task, high, low):
        cells = []
        for metric in METRICS:
            row = pairs[model, task, high, low, metric]
            cells.append(numeric(row["mean_delta_pp"], signed=True) + "［"
                         + numeric(row["ci_low_pp"], signed=True) + ", "
                         + numeric(row["ci_high_pp"], signed=True) + "］")
        return " / ".join(cells)

    return ["主要观察", "",
            f"AlphaRAD相对P&M，两指标区间同时严格高于零的任务数为单视图{alpha_pm[0]['both_ci_positive']}/3、双视图{alpha_pm[1]['both_ci_positive']}/3；"
            f"相对线性探针则为{alpha_lp[0]['both_ci_positive']}/3、{alpha_lp[1]['both_ci_positive']}/3。全部七个对手的结果见第四部分，当前结果不足以概括为普遍优于全部竞争方法。",
            "AlphaRAD的Q−Original AUPRC区间包含零的任务为：" + "、".join(zero_tasks)
            + "。相对某个适配方法的优势与相对原始模型的提升是不同比较。",
            "BiomedCLIP呈任务依赖：CSLV单视图Q−P&M的AUROC / AUPRC差值及区间为"
            + contrast("BiomedCLIP", "slvh", "query", "project_mix")
            + "；CDLV单视图Q−P&M为" + contrast("BiomedCLIP", "dlv", "query", "project_mix")
            + "，但Q−线性探针为" + contrast("BiomedCLIP", "dlv", "query", "linear_probe") + "。",
            "AlphaRAD的其他对手亦有反例：CDLV双视图Q+V−GDA为"
            + contrast("AlphaRAD", "dlv", "query_view", "gda_view")
            + "；CCOM双视图Q+V−Tip-Adapter为"
            + contrast("AlphaRAD", "composite_slvh_dlv", "query_view", "tip_adapter_view")
            + "。上述均为百分点，保留负向和指标方向不一致的结果。", ""]


def render_report(summary, fits, caches, independent, metadata):
    means, pairs, seed_pairs = indexed_rows(summary)
    lines = ["CheXchoNet 同支持、同视图近邻方法比较：完整结果分析", "", *interpretation(pairs),
             "一、范围与读数", "",
             "本轮固定 BiomedCLIP、AlphaRAD 两个底座，CSLV、CDLV、CCOM 三个二元任务，支持种子42–46。",
             "固定测试为7,407张图、2,470名患者；每任务每种子使用5名阳性和5名严格阴性支持患者各一张图。支持／测试患者隔离。",
             "每任务／种子共用5折支持患者交叉验证，每折验证1正1负；各方法均仅以原图支持拟合和选参，选定同一参数后用于单视图和双视图预测。",
             "选参使用合并折外预测的梯形AUPRC。Q固定η=0.3；双视图为原图与中心裁剪放大图，裁剪边长比例0.8，输出等权融合。",
             "本次18个方法／视图条件、2底座×3任务×5种子，共540个任务×种子×分支结果全部保留。三标签共用测试患者，不能作为三个独立队列。",
             "所有分数为五次独立适配后指标的均值±样本标准差，不是平均预测构造的集成。AUROC与梯形AUPRC单位%，差值及标准差单位百分点；AP另保存在完整结果中。",
             "95%区间由2,000次共享患者级配对bootstrap计算，条件于当前五组固定支持，不包含未来支持抽样不确定性，未经多重比较校正。",
             "CheXchoNet已用于此前整体适配及路径机制评估。本轮先固定比较方法、支持折和候选范围，再保存全部拟合／预测后评分；数据复用历史仍然保留。",
             "下表同时保留下降、混合变化、区间跨零和零增益；不得将事后挑选的模型／任务称为预设独立验证。六组合以外不作外推，不计算跨底座合并均值。", "",
             "二、六个模型×任务组合：Original、Q、P&M、线性探针", "",
             "单元格为 AUROC均值±SD / AUPRC均值±SD。单／双视图各自使用对应Original；双视图查询行为Q+V。", "",
             "| 底座 | 任务 | 视图 | Original | Q或Q+V | P&M | 线性探针 |",
             "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for task in TASKS:
            for suffix, view in (("", "单视图"), ("_view", "双视图")):
                cells = [" / ".join(mean_cell(means[model, task, branch + suffix, metric]) for metric in METRICS)
                         for branch in ("original", "query", "project_mix", "linear_probe")]
                lines.append("| " + " | ".join((model, TASK_NAMES[task], view, *cells)) + " |")
    lines.extend(["", "三、同视图Q−P&M、Q−线性探针：完整配对结果", "",
                  "每格依次给出平均差值、95%区间及五支持种子的正／负／零差值次数；双指标同时胜出的种子另列。种子胜次不等于每种子显著。", "",
                  "| 底座 | 任务 | 比较 | ΔAUROC及区间／种子次数 | ΔAUPRC及区间／种子次数 | 双指标均正种子 |",
                  "|---|---|---|---|---|---|"])
    for model in MODELS:
        for task in SCOPES:
            for suffix, label in (("", "Q"), ("_view", "Q+V")):
                for competitor in ("project_mix", "linear_probe"):
                    high, low = "query" + suffix, competitor + suffix
                    cells = [pair_cell(pairs[model, task, high, low, metric]) for metric in METRICS]
                    compare = label + " − " + NAMES[competitor] + ("（双视图）" if suffix else "（单视图）")
                    wins = joint_seed_wins(seed_pairs, model, task, high, low)
                    lines.append("| " + " | ".join((model, TASK_NAMES[task], compare, *cells, f"{wins}/5")) + " |")
    lines.extend(["", "宏平均先在每个种子内对三个任务的指标或差值等权平均，再计算五种子均值／SD及共享患者抽样区间；该行不增加独立任务数。", "",
                  "四、全部七个竞争方法：按底座与视图统计", "",
                  "每行只统计三个任务。双升／双降指Q或Q+V相对该竞争方法的两项平均指标；CI双正／双负指两项区间都严格在零同侧。混合方向、零差值和其余CI均保留。", "",
                  "| 底座 | 视图 | 竞争方法 | 双升 | 双降 | 混合 | 含零差值 | CI双正 | CI双负 | 其余CI |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
    count_keys = ("both_positive", "both_negative", "mixed", "zero_or_tie", "both_ci_positive", "both_ci_negative", "other_ci")
    for model in MODELS:
        for suffix, view in (("", "单视图"), ("_view", "双视图")):
            total = Counter({key: 0 for key in count_keys})
            for competitor in COMPETITORS:
                counts = both_counts(pairs, model, TASKS, "query" + suffix, competitor + suffix)
                total.update(counts)
                lines.append("| " + " | ".join((model, view, NAMES[competitor], *(str(counts[k]) for k in count_keys))) + " |")
            lines.append("| " + " | ".join((model, view, "合计21个相关比较", *(str(total[k]) for k in count_keys))) + " |")
    lines.extend(["", "计数仅描述本底座本视图的3任务×7对手覆盖，不作为21个独立试验计算显著性；不同对手和任务共享测试患者。", "",
                  "五、实现范围、文本秩与支持内选择", "",
                  "P&M依据论文v2公式5/8/9/10重实现，并移植为每任务正／负文本两类；公开仓库当时只有说明与许可证，因此不称为作者可执行代码的精确复现。",
                  "其文本SVD以99%累计平方奇异值能量确定秩，不通过测试成绩挑选；λ、γ各使用0至1等距10点候选。线性探针为L2逻辑回归，C为10^−4至10^4的9个十进位候选。",
                  "P&M不做梯度训练；线性探针与SimNL拟合小型参数，预训练骨干始终冻结。Tip-Adapter、GDA与SimNL仍为已记录的二元OvR移植。",
                  "BiomedCLIP查询和竞争方法使用共享全局投影空间。AlphaRAD的Q使用原始查询相关的类局部支持统计，竞争分数方法使用冻结全局投影视觉特征；标签和视图预算相同不代表中间表征相同。",
                  "AlphaRAD原生双视图先平均logit再sigmoid；BiomedCLIP原生双视图平均概率；竞争方法平均各自margin/logit。不同分数域的等权平均并不等价。", "",
                  "| 底座 | 任务 | P&M全文本秩（种子42–46） | 留存能量范围 | 第一奇异值范围 | 第二奇异值范围 |",
                  "|---|---|---|---|---|---|"])
    for model in MODELS:
        for task in TASKS:
            audits = [fits[model][seed][task]["methods"]["project_mix"]["full_support_fits"][0] for seed in SEEDS]
            ranks = [a["text_rank"] for a in audits]
            energy = [a["text_energy_retained"] for a in audits]
            singular = [[a["text_singular_values"][i] for a in audits] for i in (0, 1)]
            span = lambda values: numeric(min(values), 6) + "～" + numeric(max(values), 6)
            lines.append("| " + " | ".join((model, TASK_NAMES[task], "/".join(map(str, ranks)), span(energy),
                                                span(singular[0]), span(singular[1]))) + " |")
            for seed, audit in zip(SEEDS, audits):
                folded = fits[model][seed][task]["methods"]["project_mix"]["fold_fits"]
                require(all(part[0]["text_rank"] == audit["text_rank"] for part in folded), "固定文本秩在支持折间变化")
    lines.append("")
    for model in MODELS:
        rank_counts = Counter(fits[model][seed][task]["methods"]["project_mix"]["full_support_fits"][0]["text_rank"]
                              for seed in SEEDS for task in TASKS)
        lines.append(model + "的15个任务×种子完整拟合：" + "；".join(f"秩{rank}为{count}项" for rank, count in sorted(rank_counts.items()))
                     + "。对应支持折的文本秩均与完整拟合一致。")
    lines.extend(["",
                  "以下列出全部30个任务×种子拟合的实际选参。同一行的参数用于单视图和双视图。OOF AUPRC仅为十张支持图上的选择记录。", "",
                  "| 底座 | 任务 | 种子 | P&M λ / γ / OOF AUPRC% | LP C / OOF AUPRC% | 原型混合α | Tip α / β | GDA α | SimNL轮数 |",
                  "|---|---|---|---|---|---|---|---|---|"])
    for model in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                methods = fits[model][seed][task]["methods"]
                pm, lp = (methods[m]["selected"] for m in ("project_mix", "linear_probe"))
                pm_cell = " / ".join((numeric(pm["lambda"], 6), numeric(pm["gamma"], 6), numeric(100 * pm["oof_macro_auprc"])))
                lp_cell = f"{lp['C']:g} / " + numeric(100 * lp["oof_macro_auprc"])
                mix = methods["prototype_mix"]["selected_parameters"]
                tip = methods["tip_adapter"]["selected_parameters"]
                gda = methods["gda"]["selected_parameters"]
                lines.append("| " + " | ".join((model, TASK_NAMES[task], str(seed), pm_cell, lp_cell,
                    f"{mix['alpha']:g}", f"{tip['alpha']:g} / {tip['beta']:g}", f"{gda['alpha']:g}",
                    str(methods["simnl"]["chosen_epoch"]))) + " |")
    lines.extend(["", "六、成本记录", "",
                  "下表为本次缓存与CPU流程的实测记录。本轮重新编码两个视图的全局特征，并重新计算原生预测端点后与历史输出核验；缓存阶段耗时还包括支持图编码、查询构造、原生分支、读写和数值核验，因此不是单独的逐方法推理延迟。CPU计时包含支持交叉验证、完整支持拟合和两个测试视图预测；双视图没有另行选参。", "",
                  "| 底座 | 特征维度 | 缓存阶段秒 | 经典四方法合计秒 | SimNL合计秒 | P&M+LP合计秒 |",
                  "|---|---|---|---|---|---|"])
    for model in MODELS:
        all_fits = [fits[model][seed][task] for seed in SEEDS for task in TASKS]
        timings = [f["timing"] for f in all_fits]
        keys = ("classic_four_methods_fit_and_predict_seconds", "simnl_fit_and_predict_seconds", "neighbor_two_methods_fit_and_predict_seconds")
        totals = [numeric(sum(t[k] for t in timings), 2) for k in keys]
        lines.append("| " + " | ".join((model, str(caches[model]["feature_dimension"]),
                                           numeric(caches[model]["elapsed_seconds"], 2), *totals)) + " |")
    lines.extend(["", "| 底座 | 方法 | 15次CV＋拟合秒合计 | 两视图预测秒合计 | 每任务可训练参数范围 |",
                  "|---|---|---|---|---|"])
    for model in MODELS:
        for method in ("project_mix", "linear_probe"):
            records = [fits[model][seed][task]["methods"][method] for seed in SEEDS for task in TASKS]
            parameters = [r["trainable_parameters"] for r in records]
            lines.append("| " + " | ".join((model, NAMES[method], numeric(sum(r["fit_seconds"] for r in records), 2),
                numeric(sum(r["prediction_seconds"] for r in records), 2), f"{min(parameters)}～{max(parameters)}")) + " |")
    lines.extend(["", "Q查询构造与原生匹配未单独计时，当前记录不足以给出完整逐方法端到端速度排名。以上CPU分组之和也不等于并行实验的总墙钟时间。", "",
                  "七、完整性与来源", "",
                  "运行目录：" + metadata["run"],
                  "原始结果：RESULTS.json；全部方法均值／标准差：MEAN_SD_COMPARISON.csv；所有预设配对区间：PAIRED_COMPARISONS.csv。",
                  "逐种子值：PER_SEED_RESULTS.csv；逐种子配对值：PER_SEED_PAIRED_COMPARISONS.csv；完整候选OOF、训练参数与文本谱：各模型evaluation/seed_*/fit_info.json。",
                  "生成器只读取已完成结果和拟合记录，不读取测试标签，不重算模型预测。完整统计保留540项分支结果；本轮不增加历史135项Q+V主表任务数。",
                  "独立统计审计状态：" + independent.get("status", "unknown")
                  + ("；详见INDEPENDENT_AUDIT.json。" if independent.get("status") == "passed" else "；当前报告不代替尚待完成的独立审计。"),
                  "RESULTS.json SHA256：" + metadata["results_sha256"],
                  "FIT_COMPLETION_AUDIT.json SHA256：" + metadata["fit_audit_sha256"],
                  "报告生成器SHA256：" + metadata["generator_sha256"],
                  "生成时间UTC：" + metadata["generated_at_utc"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = render_report(*complete_inputs(run))
    path = run / "ANALYSIS.txt"
    temporary = path.with_suffix(".txt.tmp")
    temporary.write_text(report)
    temporary.replace(path)
    print(json.dumps(dict(status="complete", path=str(path), sha256=sha(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
