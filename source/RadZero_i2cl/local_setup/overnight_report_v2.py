#!/usr/bin/env python3
"""Report all v2 experiments and the user's stronger, unrounded success criterion."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from overnight_report import PROJECT, DATASETS, METRICS, collect, read, write


def rank_families(rows):
    groups = {}
    for row in rows:
        key = row["family"], row["dataset"]
        if key in groups:
            raise ValueError(f"Duplicate family/dataset: {key}")
        groups[key] = row
    ranking = []
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        all_five = len(group) == 5 and all(r["both_above"] for r in group)
        one_point = sum(r["delta_auroc_pp"] >= 1.0 and r["delta_auprc_pp"] >= 1.0
                        for r in group)
        two_point = sum(r["delta_auroc_pp"] >= 2.0 and r["delta_auprc_pp"] >= 2.0
                        for r in group)
        ranking.append({
            "family": family, "completed_datasets": len(group),
            "datasets_both_above": sum(r["both_above"] for r in group),
            "all_five_both_above": all_five,
            "datasets_both_ge_1pp": one_point, "datasets_both_ge_2pp": two_point,
            "enhanced_target_met": all_five and one_point >= 4,
            "mean_delta_auroc_pp": sum(r["delta_auroc_pp"] for r in group) / len(group),
            "mean_delta_auprc_pp": sum(r["delta_auprc_pp"] for r in group) / len(group),
        })
    ranking.sort(key=lambda r: (
        -r["enhanced_target_met"], -r["all_five_both_above"],
        -r["completed_datasets"], -r["datasets_both_ge_1pp"],
        -r["datasets_both_above"], -r["mean_delta_auroc_pp"],
        -r["mean_delta_auprc_pp"], r["family"]))
    return ranking


def collect_v2(campaign, raw):
    rows, _, errors, job_states = collect(campaign, raw)
    reference = {
        name: read(PROJECT / "local_assets/outputs/public5_posweighted_20261020_2014"
                   / name / "baseline_metrics.json")
        for name in DATASETS
    }

    def add(family, name, record, source, selection):
        metrics, base = record["metrics"], record["baseline_metrics"]
        if any(abs(base[k] - reference[name][k]) > 1e-12 for k in METRICS):
            raise ValueError(f"Baseline identity failure: {family}/{name}")
        rows.append({
            "family": family, "dataset": name, "selection": selection,
            "config": record.get("config", {}), "source": str(source),
            "auroc": metrics["macro_auroc"], "auprc": metrics["macro_auprc"],
            "baseline_auroc": base["macro_auroc"], "baseline_auprc": base["macro_auprc"],
            "delta_auroc_pp": 100 * (metrics["macro_auroc"] - base["macro_auroc"]),
            "delta_auprc_pp": 100 * (metrics["macro_auprc"] - base["macro_auprc"]),
            "both_above": all(metrics[k] > base[k] for k in METRICS),
        })

    branches = ("ridge_wide", "ridge_perclass", "weighted_kernel",
                "ranking_residual", "query_ridge_oof")
    for branch in branches:
        for name in DATASETS:
            directory = raw / branch / name
            source = directory / "summary.json"
            try:
                if (directory / "status.json").exists():
                    job_states.append({"branch": branch, "dataset": name,
                                       **read(directory / "status.json")})
                if not source.exists():
                    continue
                summary = read(source)
                for index, record in enumerate(summary["results"]):
                    config = record.get("config", {})
                    if branch == "ridge_wide":
                        family = f"{branch}:{config['feature']}"
                    elif branch == "weighted_kernel":
                        family = f"{branch}:{record.get('family', config.get('family'))}"
                        if family.endswith(":None"):
                            raise ValueError("Kernel family not specified")
                    else:
                        family = branch
                    add(family, name, record, source, "grouped support OOF only")
                    if index == 0 and branch in ("ridge_wide", "weighted_kernel"):
                        add(f"{branch}:support_selected", name, record, source,
                            "feature/family and hyperparameters: grouped support OOF only")
            except Exception as error:
                errors.append({"source": str(source), "error": repr(error)})
    for name in DATASETS:
        source = (PROJECT / "local_assets/outputs/overnight_20261020_2150"
                  / "cache_loss_tiebreak" / name / "summary.json")
        try:
            for record in read(source)["results"]:
                config = record["config"]
                if config["method"] == "ridge" and config["feature"] == "class_local":
                    add("previous_selected_class_local_ridge", name, record, source,
                        "support OOF hyperparameters; family chosen in previous exploratory campaign")
        except Exception as error:
            errors.append({"source": str(source), "error": repr(error)})
    for source in sorted(campaign.glob("*controller/state.json")):
        if source.parent.name == "controller":
            continue  # Included by the original collector.
        try:
            state = read(source)
            jobs = state.get("jobs", {})
            entries = jobs.items() if isinstance(jobs, dict) else (
                (str(i), job) for i, job in enumerate(jobs))
            for job_id, job in entries:
                job_states.append({
                    "controller": source.parent.name, "job_id": job_id,
                    "status": job.get("status"), "returncode": job.get("returncode"),
                    "error": job.get("error", job.get("reason")),
                    "log": job.get("log"), "source": str(source),
                })
        except Exception as error:
            errors.append({"source": str(source), "error": repr(error)})
    return rows, rank_families(rows), errors, job_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path,
                        default=PROJECT / "runs/overnight_v2_20261020_2250")
    parser.add_argument("--raw", type=Path,
                        default=PROJECT / "local_assets/outputs/overnight_v2_20261020_2250")
    args = parser.parse_args()
    rows, ranking, errors, jobs = collect_v2(args.campaign, args.raw)
    limitation = (
        "探索性方法开发：测试反馈已用于决定下一步方法，不能视为未触碰的最终测试。"
        "每个分支的支持集选参和固定网格比较分别注明；不按测试标签拼接各类结果。"
        "原始 RadZero 为零样本，本实验使用固定 seed42 的有标注支持集。"
        "AUPRC 是梯形 PR 面积，不是 AP；数值达标仍需独立审计。")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory_progress",
        "target": "All 5 both improve; at least 4 both >=1pp, using unrounded values",
        "limitation": limitation, "rows": rows, "ranking": ranking,
        "errors": errors, "job_states": jobs,
    }
    write(args.campaign / "leaderboard.json", report)
    path = args.campaign / "leaderboard.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    lines = [
        f"RadZero 第二轮夜间实验 — {report['generated_at']}",
        "截止：2026-10-21 10:00 北京时间；仅使用空闲且取得租约的 GPU3–7。",
        "目标：五集 AUROC/AUPRC 均提升，至少四集两项均提升 ≥1 个百分点。",
        limitation, "",
        "方法 | 完成数据集 | 双指标提升 | 双指标各≥1点 | 数值达标（待审计）",
    ]
    for r in ranking:
        lines.append(f"{r['family']} | {r['completed_datasets']}/5 | "
                     f"{r['datasets_both_above']}/5 | {r['datasets_both_ge_1pp']}/5 | "
                     f"{r['enhanced_target_met']}")
    lines += ["", "各候选结果（指标为百分数，增量为百分点）",
              "方法 | 数据集 | AUROC | AUPRC | ΔAUROC | ΔAUPRC"]
    for row in rows:
        lines.append(f"{row['family']} | {row['dataset']} | {100*row['auroc']:.5f} | "
                     f"{100*row['auprc']:.5f} | {row['delta_auroc_pp']:+.5f} | "
                     f"{row['delta_auprc_pp']:+.5f}")
    lines += ["", "任务记录（包含失败、等待和中止）："]
    for job in jobs:
        identity = job.get("job_id", f"{job.get('branch')}/{job.get('dataset')}")
        lines.append(f"{identity}: {job.get('status')}; error={job.get('error')}")
    if errors:
        lines += ["", "报告读取错误：", json.dumps(errors, ensure_ascii=False, indent=2)]
    (args.campaign / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "families": len(ranking),
                      "enhanced_target_families": [r["family"] for r in ranking
                                                   if r["enhanced_target_met"]],
                      "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
