#!/usr/bin/env python3
"""Render only completed, audited query-path results as a Chinese supplement."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEDICAL = ROOT.parents[1]
PAPER = MEDICAL / "baselines/paper_draft_zh"
RUNS = [ROOT / "runs/diagnostic_20261024_v1", ROOT / "runs/siim_holdout_20261024_v1"]
NAMES = {"rsna_pneumonia": "RSNA（既有开发集）", "chestxdet10": "ChestXDet10（既有开发集）",
         "siim_new_holdout": "SIIM（新增同源留出）"}
METHODS = {"original": "Original", "readout_only": "仅改读出 R", "aggregation_only": "仅改聚合 A",
           "query": "完整 Q", "score_retained_original_map": "S＋保留 Original 空间图"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def mean(row, metric):
    return f"{float(row[metric+'_mean_percent']):.2f} ± {float(row[metric+'_sd_pp']):.2f}"


def ci(row, metric):
    return f"{float(row[metric+'_delta_pp']):+.2f}［{float(row[metric+'_ci_low_pp']):+.2f}, {float(row[metric+'_ci_high_pp']):+.2f}］"


def main():
    sources, means, pairs, spaces, space_pairs = {}, [], [], [], []
    for run in RUNS:
        folder = run / "evaluation"
        result, audit = json.loads((folder / "RESULTS.json").read_text()), json.loads((folder / "AUDIT.json").read_text())
        if result["status"] != "complete" or audit["status"] != "passed":
            raise ValueError("All full runs and independent audits must complete first")
        for filename in ("RESULTS.json", "AUDIT.json", "MEAN_SD_COMPARISON.csv", "PAIRED_COMPARISONS.csv",
                         "SPATIAL_MEAN_SD_COMPARISON.csv", "SPATIAL_PAIRED_COMPARISONS.csv"):
            path = folder / filename
            sources[str(path)] = sha(path)
        means += rows(folder / "MEAN_SD_COMPARISON.csv")
        pairs += rows(folder / "PAIRED_COMPARISONS.csv")
        spaces += rows(folder / "SPATIAL_MEAN_SD_COMPARISON.csv")
        space_pairs += rows(folder / "SPATIAL_PAIRED_COMPARISONS.csv")
    get_pair = lambda ds, ref="readout_only": next(r for r in pairs if r["dataset"] == ds and r["target"] == "query" and r["reference"] == ref)
    get_space = lambda ds, ref="readout_only": next(r for r in space_pairs if r["dataset"] == ds and r["target"] == "query" and r["reference"] == ref)
    lead = ["# 原生查询聚合／读出路径干预与新增同源留出验证", "",
        "以下均为本轮实际运行结果，由未舍入结果文件生成；不采用假设评审稿中的区间。三个范围分别报告，不并入既有135项Q+V主表任务。", "",
        "冻结AlphaRAD，固定单视图、η＝0.3、支持种子42–46，令 q 为原查询、q′为同支持差分计算的修正查询。", "",
        "| 分支 | 注意力／视觉聚合查询 | 最终分类读出查询 |",
        "|---|---|---|", "| Original | q | q |", "| 仅改读出 R | q | q′ |",
        "| 仅改聚合 A | q′ | q |", "| 完整 Q | q′ | q′ |", "",
        "主要比较为Q−R，直接检验重新聚合在修改读出之外的贡献。S沿用同类局部特征上的最终分数残差，按支持集交叉验证选权重，不能与R等同。RSNA／ChestXDet10的S复用已核验同五组支持结果；新SIIM的S重新拟合。", "",
        "## 三个范围的主要分类对照", "",
        "| 数据范围 | Q−R ΔAUROC［95% CI］ | Q−R ΔAUPRC［95% CI］ |", "|---|---:|---:|"]
    for ds in NAMES:
        row = get_pair(ds)
        lead.append(f"| {NAMES[ds]} | {ci(row, 'auroc')} | {ci(row, 'auprc')} |")
    lead += ["", "增益单位为百分点。分类采用类别宏平均AUROC和PR曲线梯形AUPRC。", "",
        "两个既有开发集的Q−R分类增益区间均高于零，但新增SIIM的两项分类差值区间均低于零。重新聚合的额外分类收益在本轮不同数据／支持协议间不一致，不能宣称它普遍优于仅改读出。SIIM的pixel AUROC亦下降，mask-PG区间跨零；这些反例与正向结果共同保留。", "",
        "RSNA使用既定5,327图／3,720名映射NIH患者；ChestXDet10为分类492图、框定位409图，缺少可靠患者标识。两者此前参与过开发。", "",
        "SIIM先排除已核验历史清单中的1,714图及其DICOM标识组、原始／转换后精确像素重复，再固定1,000张测试图（211阳性）；其余候选按五种子各取5阳性、5阴性支持。该清单在新推理前冻结。PatientID为每图唯一UUID，缺少跨检查身份映射；只称新增同源留出、DICOM标识组隔离，不称已核实患者隔离或新医院外部验证，也未完成预训练患者重叠审计。", ""]
    for ds in NAMES:
        metrics = ["auroc", "auprc"] + (["pg"] if ds == "chestxdet10" else ["pixel_auroc", "mask_pg"] if ds == "siim_new_holdout" else [])
        titles = ["AUROC", "AUPRC"] + (["框 PG"] if ds == "chestxdet10" else ["阳性图 pixel AUROC", "mask PG"] if ds == "siim_new_holdout" else [])
        lead += [f"## {NAMES[ds]}：全部分支与预设Q对照", "", "| 方法／比较 | " + " | ".join(titles) + " |",
                 "|---|" + "---:|"*len(metrics)]
        ds_space = {r["method"]:r for r in spaces if r["dataset"] == ds}
        for row in (r for r in means if r["dataset"] == ds):
            values = [mean(row, m) for m in metrics[:2]]
            values += [mean(ds_space[row["method"]], m) for m in metrics[2:]]
            lead.append("| " + " | ".join([METHODS[row["method"]], *values]) + " |")
        for ref in ("original", "readout_only", "aggregation_only", "score_retained_original_map"):
            row = get_pair(ds, ref)
            values = [ci(row, m) for m in metrics[:2]]
            values += [ci(get_space(ds, ref), m) for m in metrics[2:]]
            lead.append("| " + " | ".join(["Q − " + METHODS[ref], *values]) + " |")
        lead += ["", "绝对指标为百分数，±为五种子样本标准差；差值行为百分点和配对95%区间。", ""]
    lead += ["## 空间指标及结论边界", "",
        "AlphaRAD原生grounding map是注意力查询与patch的逐头scaled cosine均值，在softmax及context聚合之前生成。R与Original地图完全相同，A与Q地图完全相同；这种相等由实现定义决定。冻结patch的dense重打分也能复现Q地图，因此空间变化不能单独证明非线性重新聚合的优势或分类解释忠实性。S保留的原始图不解释新增分数项。", "",
        "ChestXDet10沿用原生框PG；SIIM去CLS后将37×37响应双线性映回1024×1024原始掩码，逐阳性图计算pixel AUROC再等权平均，并计算峰值落入并集掩码的mask-PG。SIIM不选阈值，不把这些指标称为Dice分割性能；空间评价只覆盖211张阳性图。", "",
        "这些实验回答具体计算路径的贡献，不能据此得出‘必须修改原生查询’或‘所有分数校正方法都无空间适配能力’。区间跨零或负向结果全部保留，未用显著性筛选对照。", "",
        "## 统计与来源", "",
        "使用2,000次配对组bootstrap、种子20261024，同数据集所有方法／支持种子共享抽样。每个种子先计算指标，再平均五种子指标；不平均预测构造集成。区间条件于当前五组支持，不包含未来支持抽样总体波动，未作多重比较校正。RSNA按映射患者，ChestXDet10按图像／精确重复组，SIIM按DICOM标识组重采样。像素不是独立重采样单位。", "",
        "[开发集完整统计](../../experiments/query_path_intervention_20261024/runs/diagnostic_20261024_v1/evaluation/RESULTS.txt)、[新增SIIM完整统计](../../experiments/query_path_intervention_20261024/runs/siim_holdout_20261024_v1/evaluation/RESULTS.txt)、[运行说明](../../experiments/query_path_intervention_20261024/RUN_LOCAL.txt)、[SIIM冻结协议](../../experiments/query_path_intervention_20261024/protocol_siim_v1/catalog.json)。", ""]
    report = PAPER / "原生查询路径干预_补充实验.md"
    report.write_text("\n".join(lead))
    insert = ["### 5.7 原生查询聚合／读出干预与新增同源留出", "",
        "为区分查询修正对最终匹配和视觉聚合的影响，我们在AlphaRAD上固定原图单视图、η＝0.3和五组支持（种子42–46），分别运行Original（聚合q／读出q）、仅改读出R（q／q′）、仅改聚合A（q′／q）和完整Q（q′／q′）。逐头归一化、温度、冻结头权重及CLS处理保持原生实现；Original和Q经过真实原生模块数值核验。主要比较Q−R检验重新聚合在修改读出之外的分类贡献。", "",
        "| 数据范围 | Q−R ΔAUROC［95% CI］ | Q−R ΔAUPRC［95% CI］ |", "|---|---:|---:|"]
    for ds in NAMES:
        row = get_pair(ds)
        insert.append(f"| {NAMES[ds]} | {ci(row, 'auroc')} | {ci(row, 'auprc')} |")
    insert += ["", "表13：五支持种子的平均配对差值，单位为百分点；2,000次配对组bootstrap的区间条件于这五组支持。RSNA按映射患者，ChestXDet10按图像／精确重复组，SIIM按DICOM标识组抽样，未作多重比较校正。", "",
        "RSNA与ChestXDet10复用既有开发测试；新增SIIM从原训练来源中排除已核验历史1,714图及相同标识组、精确像素重复后，固定1,000张测试图（211阳性），五组支持各含5阳性、5阴性。支持和测试在新推理前冻结，不使用测试结果选择分支。SIIM的每图UUID未获可靠跨检查患者映射，故本实验属于新增同源留出，不能称为已核实患者隔离的外部队列，预训练重叠亦未完成审计。", "",
        f"空间上，ChestXDet10的Q−R框PG为{ci(get_space('chestxdet10'), 'pg')}个百分点。新增SIIM的Q−R阳性图pixel AUROC为{ci(get_space('siim_new_holdout'), 'pixel_auroc')}、mask-PG为{ci(get_space('siim_new_holdout'), 'mask_pg')}个百分点；它们基于原图尺度真实掩码，不进行阈值选择，也不等同于Dice分割性能。", "",
        "原生grounding map来自注意力查询与patch的逐头scaled cosine，在softmax和context聚合之前生成。因此R与Original空间图相同，A与Q空间图相同；dense特征重打分亦可复现Q图。空间输出变化不能单独证明非线性聚合或分类解释忠实性。", "",
        "两既有开发集支持重新聚合带来额外分类收益，但新增SIIM的Q−R两项分类差值区间均低于零，pixel AUROC也下降，mask-PG差值仍不确定。本轮不能将查询重聚合表述为普遍更优或‘必须修改原生查询’；证据支持的是效果有条件的适配接口及其明确失败情形。", "",
        "全部分支的均值、标准差、Q−A、Q−S及指标尺度交互见[补充实验完整表](原生查询路径干预_补充实验.md)，实现与复跑入口见[运行说明](../../experiments/query_path_intervention_20261024/RUN_LOCAL.txt)。本补充不与既有135项Q+V任务混算。", ""]
    (ROOT / "MANUSCRIPT_INSERT.md").write_text("\n".join(insert))
    audit = dict(status="rendered_from_completed_results", sources=sources,
        report=str(report), report_sha256=sha(report), insert_sha256=sha(ROOT / "MANUSCRIPT_INSERT.md"),
        no_hypothetical_results=True, script_sha256=sha(Path(__file__)))
    (ROOT / "SUPPLEMENT_BUILD_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    print(report)


if __name__ == "__main__":
    main()
