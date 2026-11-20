#!/usr/bin/env python3
"""Build the bilingual main classification table from saved, unrounded results."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


PAPER = Path(__file__).resolve().parents[1]
MODELS = [
    "AlphaRAD", "RadZero", "BiomedCLIP", "BioViL-T", "CheXzero",
    "MedKLIP", "MAVL", "KAD", "MaCo", "CARZero",
]
MODEL_REFERENCES = {
    "AlphaRAD": "alpharad", "RadZero": "radzero", "BiomedCLIP": "biomedclip",
    "BioViL-T": "biovilt", "CheXzero": "chexzero", "MedKLIP": "medklip",
    "MAVL": "mavl", "KAD": "kad", "MaCo": "maco", "CARZero": "carzero",
}
BRANCHES = [("original", "Original"), ("tta_center_refined", "Q+V")]
ALL_BRANCHES = [
    ("original", "Original"), ("query_refined", "Q"),
    ("tta_center_original", "V"), ("tta_center_refined", "Q+V"),
]
METRICS = ["macro_auroc", "macro_auprc"]
PANELS = [
    [
        ("chest14", "ChestXray14"), ("chestxdet10", "ChestXDet10"),
        ("rsna_pneumonia", "RSNA"), ("siim_pneumothorax", "SIIM (1,704)"),
        ("siim_radzero_holdout", "SIIM (1,694)"), ("chexpert", "CheXpert"),
    ],
    [
        ("shenzhenxray", "Shenzhen"),
        ("montgomery", "Montgomery"), ("covidkaggle", "CovidKaggle"),
        ("chexchonet_slvh", "CSLV"), ("chexchonet_dlv", "CDLV"),
        ("chexchonet_composite_slvh_dlv", "CCOM"),
    ],
]
TABLE_AVERAGE_GROUPS = {}
for panel in PANELS:
    for dataset, _ in panel:
        group = "chexchonet" if dataset.startswith("chexchonet_") else dataset
        TABLE_AVERAGE_GROUPS.setdefault(group, []).append(dataset)

CAPTIONS = {
    "en": (
        r"\textbf{Main classification results: Original versus our full method (Q+V).} "
        r"Each entry is AUROC / AUPRC (\%) under fixed support. The higher value "
        r"for each metric within each model and column is bolded "
        r"(ties at the displayed precision are both bolded). CheXchoNet tasks and "
        r"SIIM protocols are listed separately. Average gives the equally weighted "
        r"mean over reported models in each column. Avg., shown only in the second panel, "
        rf"equally averages the {len(TABLE_AVERAGE_GROUPS)} dataset/protocol groups across both panels: the three "
        r"CheXchoNet tasks are averaged first, and both SIIM protocols count separately. "
        r"All means are computed before rounding. "
        r"NA: required task interface or query resources unavailable; "
        r"---: not evaluated. Neither is included in averages."
    ),
    "zh": (
        r"\textbf{分类主结果：Original 与本文完整方法 Q+V 的比较。}"
        r"每项为固定支持集下的 AUROC / AUPRC（\%）。"
        r"同一模型、同一列内，各指标较高值加粗（按显示精度并列时均加粗）。"
        r"CheXchoNet 的三个任务及 SIIM 的两种协议分别列出。"
        r"平均行对每列已报告的模型取等权平均。Avg. 仅列于第二个子表末尾，"
        rf"对两个子表中的 {len(TABLE_AVERAGE_GROUPS)} 个数据集或协议组取等权平均：CheXchoNet 的三个任务先取平均，"
        r"SIIM 的两种协议分别计入。所有均值均使用未四舍五入的数值计算。"
        r"NA：缺少所需任务接口或查询资源；---：未评价。二者均不计入平均。"
    ),
}


def read_results():
    with (PAPER / "supplement_data/fixed_support_all.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    values = {}
    for row in rows:
        if row["task"] != "classification":
            continue
        key = tuple(row[field] for field in ("model", "dataset", "branch", "metric"))
        assert key not in values, key
        values[key] = float(row["score_percent"])
    verified = json.loads(
        (PAPER.parent / "baselines/paper_draft_zh/verified_results.json").read_text()
    )
    records = verified["records"]
    assert verified["included_models"] == MODELS
    assert {key[0] for key in values} == set(MODELS)
    statuses = {
        (row["model"], row["dataset"]): row["status"]
        for row in records if row["task"] == "classification"
    }
    completed = {key[:2] for key in values}
    assert len(completed) == verified['summary']['complete_task_counts']['classification']
    assert completed == {key for key, value in statuses.items() if value == "complete"}
    assert {key[1] for key in completed} == {ds for panel in PANELS for ds, _ in panel}
    for model, dataset in completed:
        for branch, _ in ALL_BRANCHES:
            for metric in METRICS:
                assert (model, dataset, branch, metric) in values
    return values, statuses


def format_pair(current, other):
    cells = []
    for value, baseline in zip(current, other):
        displayed = f"{value:.2f}"
        if float(displayed) >= float(f"{baseline:.2f}"):
            displayed = r"\textbf{" + displayed + "}"
        cells.append(displayed)
    return " / ".join(cells)


def table_means(values, statuses):
    """Average both panels, grouping cardiac tasks and retaining both SIIM protocols."""
    summaries = {}
    for model in MODELS:
        groups = [datasets for datasets in TABLE_AVERAGE_GROUPS.values()
                  if all(statuses.get((model, ds)) == "complete" for ds in datasets)]
        if not groups:
            continue
        summaries[model] = {
            branch: {metric: mean(mean(values[model, ds, branch, metric] for ds in datasets)
                                  for datasets in groups)
                     for metric in METRICS}
            for branch, _ in BRANCHES
        }
    summaries["Average"] = {
        branch: {metric: mean(scores[branch][metric] for scores in summaries.values())
                 for metric in METRICS}
        for branch, _ in BRANCHES
    }
    return summaries


def render(language, values, statuses):
    model_label, branch_label, average_label = (
        ("Model", "Branch", "Average") if language == "en" else ("模型", "分支", "平均")
    )
    averages = defaultdict(list)
    table_summary = table_means(values, statuses)
    for (model, dataset, branch, metric), value in values.items():
        averages[(dataset, branch, metric)].append(value)

    def cell(model, dataset, branch):
        other = "tta_center_refined" if branch == "original" else "original"
        if model is None:
            current = [mean(averages[(dataset, branch, metric)]) for metric in METRICS]
            comparison = [mean(averages[(dataset, other, metric)]) for metric in METRICS]
        else:
            status = statuses.get((model, dataset))
            if status != "complete":
                assert status in (None, "unsupported"), (model, dataset, status)
                return "NA" if status == "unsupported" else "---"
            current = [values[(model, dataset, branch, metric)] for metric in METRICS]
            comparison = [values[(model, dataset, other, metric)] for metric in METRICS]
        return format_pair(current, comparison)

    caption = CAPTIONS[language]
    if all(statuses.get((model, ds)) == 'complete'
           for model in MODELS for panel in PANELS for ds, _ in panel):
        caption = caption.replace(
            r"NA: required task interface or query resources unavailable; ---: not evaluated. Neither is included in averages.", ""
        ).replace(r"NA：缺少所需任务接口或查询资源；---：未评价。二者均不计入平均。", "").rstrip()
    elif all((model, ds) in statuses for model in MODELS for panel in PANELS for ds, _ in panel):
        caption = caption.replace(
            r"; ---: not evaluated. Neither is included in averages.",
            r". Unavailable entries are excluded from averages.",
        ).replace(r"；---：未评价。二者均不计入平均。", r"，不计入平均。")
    lines = [
        "% Generated by scripts/build_classification_table.py; edit the generator.",
        r"\begin{table*}[tp]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:overall}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
    ]
    for panel_index, panel in enumerate(PANELS):
        if panel_index:
            lines.append(r"\par\vspace{6pt}")
        show_avg = panel_index == 1
        columns = "ll" + r">{\centering\arraybackslash}X" * (len(panel) + int(show_avg))
        lines += [
            r"\begin{tabularx}{\textwidth}{" + columns + "}",
            r"\toprule",
        ]
        if panel_index == 0:
            lines.append(" & ".join([model_label, branch_label] + [label for _, label in panel]) + r" \\")
        else:
            cardiac_start = next(i for i, (ds, _) in enumerate(panel) if ds.startswith("chexchonet_"))
            cardiac_columns = panel[cardiac_start:]
            assert all(ds.startswith("chexchonet_") for ds, _ in cardiac_columns)
            first_column = cardiac_start + 3
            last_column = first_column + len(cardiac_columns) - 1
            lines += [
                " & ".join([model_label, branch_label] + [label for _, label in panel[:cardiac_start]]
                           + [rf"\multicolumn{{{len(cardiac_columns)}}}{{c}}{{CheXchoNet}}", "Avg."]) + r" \\",
                rf"\cmidrule(lr){{{first_column}-{last_column}}}",
                " & ".join([""] * (cardiac_start + 2) + [label for _, label in cardiac_columns] + [""]) + r" \\",
            ]
        lines.append(r"\midrule")
        for model_index, model in enumerate(MODELS + [None]):
            if model is None:
                lines.append(r"\midrule")
            elif model_index:
                lines.append(r"\addlinespace[1.5pt]")
            for branch_index, (branch, label) in enumerate(BRANCHES):
                if branch == "tta_center_refined":
                    lines.append(r"\rowcolor{qvblue}")
                name = (model or average_label) if branch_index == 0 else ""
                if model is not None and branch_index == 0:
                    name += rf"~\cite{{{MODEL_REFERENCES[model]}}}"
                cells = [name, label] + [cell(model, ds, branch) for ds, _ in panel]
                if show_avg:
                    scores = table_summary.get(model or "Average")
                    other = "tta_center_refined" if branch == "original" else "original"
                    avg = format_pair([scores[branch][metric] for metric in METRICS],
                                      [scores[other][metric] for metric in METRICS]) if scores else "---"
                    cells.append(avg)
                lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabularx}"]
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def component_means(values):
    grouped = defaultdict(list)
    for (model, dataset, branch, metric), score in values.items():
        if dataset == "siim_radzero_holdout":
            continue
        dataset = "chexchonet" if dataset.startswith("chexchonet_") else dataset
        grouped[(model, dataset, branch, metric)].append(score)
    dataset_means = {key: mean(scores) for key, scores in grouped.items()}
    summaries = {}
    for model in MODELS:
        datasets = sorted({key[1] for key in dataset_means if key[0] == model})
        summaries[model] = {
            "dataset_count": len(datasets),
            "means": {branch: {metric: mean(dataset_means[model, ds, branch, metric] for ds in datasets)
                               for metric in METRICS} for branch, _ in ALL_BRANCHES},
        }
    overall = {branch: {metric: mean(summaries[model]["means"][branch][metric] for model in MODELS)
                        for metric in METRICS} for branch, _ in ALL_BRANCHES}
    pairs = {(model, dataset) for model, dataset, _, _ in dataset_means}
    effect = {metric: mean(dataset_means[model, ds, "tta_center_refined", metric]
                           - dataset_means[model, ds, "query_refined", metric]
                           for model, ds in pairs) for metric in METRICS}
    return summaries, overall, effect


def render_components(language, summaries, overall):
    if language == "en":
        caption = (
            r"\textbf{Component ablation of the full Q+V method.} Each entry is AUROC / AUPRC (\%). "
            r"$N$ is the number of datasets averaged for that model, with the cardiac tasks grouped and "
            r"the same 1,704-image SIIM protocol retained for all models. "
            r"The last row equally weights the ten model means, computed before rounding. "
            r"Original: baseline; Q: without view fusion; V: without query adaptation; Q+V: full method (Sec.~\ref{sec:views}). "
            r"The highest value for each metric in each row is bolded. All are fixed-support point estimates. "
            r"The comparison assesses adaptation within each backbone."
        )
        model_label, average_label = "Backbone", "Average"
    else:
        caption = (
            r"\textbf{完整方法 Q+V 的组件消融实验。}每项为 AUROC / AUPRC（\%）。"
            r"$N$ 为该模型求平均时纳入的数据集数，其中心脏任务合并，所有模型统一采用 SIIM 的 "
            r"1,704 张图像协议。末行对十个模型的均值等权平均，"
            r"使用未四舍五入的数值计算。Original：基线；Q：移除视图融合；V：移除查询适配；Q+V：完整方法"
            r"（见第~\ref{sec:views} 节）。每行各指标的最高值加粗。所有结果均为固定支持集下的点估计。"
            r"该比较用于分析各底座内部的适配效果。"
        )
        model_label, average_label = "底座", "平均"
    lines = [
        "% Generated by scripts/build_classification_table.py; edit the generator.",
        r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}",
        r"\label{tab:components}", r"\small",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrcccc}", r"\toprule",
        model_label + r" & $N$ & Original & Q & V & \textbf{Q+V} \\", r"\midrule",
    ]
    for model in MODELS + [None]:
        if model is None:
            lines.append(r"\midrule")
            name, count, scores = average_label, "---", overall
        else:
            name, count, scores = model, str(summaries[model]["dataset_count"]), summaries[model]["means"]
        maxima = {metric: max(float(f'{scores[b][metric]:.2f}') for b, _ in ALL_BRANCHES) for metric in METRICS}
        cells = [name, count]
        for branch, _ in ALL_BRANCHES:
            cells.append(format_pair([scores[branch][metric] for metric in METRICS],
                                     [maxima[metric] for metric in METRICS]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    values, statuses = read_results()
    summaries, overall, effect = component_means(values)
    verified = json.loads((PAPER.parent / "baselines/paper_draft_zh/verified_results.json").read_text())
    for model in MODELS:
        expected = verified["summary"]["classification_means_unique_datasets"][model]
        assert summaries[model]["dataset_count"] == expected["count"] == 9
        for branch, _ in ALL_BRANCHES:
            for metric in METRICS:
                assert abs(summaries[model]["means"][branch][metric]
                           - 100 * expected["means"][branch][metric]) < 1e-10
    for language, suffix in [("en", ""), ("zh", "_zh")]:
        destination = PAPER / f"sec/classification_comparison{suffix}.tex"
        destination.write_text(render(language, values, statuses))
        count = len({key[:2] for key in values})
        print(f"Wrote {destination.relative_to(PAPER)}: {count} comparisons, {sum(len(panel) for panel in PANELS)} column averages and one Avg. column over {len(TABLE_AVERAGE_GROUPS)} groups across both panels.")
        (PAPER / f"sec/component_ablations{suffix}.tex").write_text(render_components(language, summaries, overall))
    audit = {"model_count": len(MODELS),
             "component_siim_protocol": {"dataset": "siim_pneumothorax", "test_images": 1704, "models": MODELS},
             "model_means_percent": summaries, "model_average_percent": overall,
             "query_plus_view_minus_query_pp": effect,
             "table2_average": {
                 "displayed_panel_index": 1,
                 "group_count": len(TABLE_AVERAGE_GROUPS),
                 "groups": TABLE_AVERAGE_GROUPS,
                 "means_percent": table_means(values, statuses),
             }}
    (PAPER / "build/CLASSIFICATION_TABLE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
