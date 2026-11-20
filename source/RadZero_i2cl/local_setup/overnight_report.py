#!/usr/bin/env python3
"""Summarize all recorded exploratory candidates without hiding unsuccessful trials."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")
METRICS = ("macro_auroc", "macro_auprc")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def collect(campaign, raw):
    rows, errors, job_states = [], [], []
    references = {}
    for name in DATASETS:
        directory = PROJECT / "local_assets/outputs/public5_posweighted_20261020_2014" / name
        references[name] = read(directory / "baseline_metrics.json")

    def add(family, name, metrics, base, source, config=None, selection="fixed"):
        for key in METRICS:
            if abs(base[key] - references[name][key]) > 1e-12:
                raise ValueError(f"Nonmatching baseline: {family}/{name}/{key}")
        rows.append({
            "family": family, "dataset": name, "selection": selection,
            "config": config or {}, "source": str(source),
            "auroc": metrics["macro_auroc"], "auprc": metrics["macro_auprc"],
            "baseline_auroc": base["macro_auroc"],
            "baseline_auprc": base["macro_auprc"],
            "delta_auroc_pp": 100 * (metrics["macro_auroc"] - base["macro_auroc"]),
            "delta_auprc_pp": 100 * (metrics["macro_auprc"] - base["macro_auprc"]),
            "both_above": all(metrics[k] > base[k] for k in METRICS),
        })

    for family, parent in [
        ("previous_weighted_i2cl", PROJECT / "local_assets/outputs/public5_posweighted_20261020_2014"),
        ("previous_regularized_i2cl", PROJECT / "local_assets/outputs/public5_identity_reg001_20261020_2120"),
        ("lr0001", raw / "lr0001"),
    ]:
        for name in DATASETS:
            source = parent / name / "result.json"
            if source.exists():
                try:
                    result = read(source)
                    if result["status"] == "complete":
                        add(family, name, result["metrics"], result["baseline_metrics"], source)
                except Exception as error:
                    errors.append({"source": str(source), "error": repr(error)})
    for family in ("blend_weighted", "blend_regularized"):
        source = campaign / "experiments" / family / "summary.json"
        if source.exists():
            try:
                for trial in read(source)["trials"]:
                    for name, result in trial["datasets"].items():
                        add(f"{family}:{trial['strength']:g}", name,
                            result["metrics"], result["baseline_metrics"], source,
                            {"strength": trial["strength"]}, "prespecified grid; exploratory test comparison")
            except Exception as error:
                errors.append({"source": str(source), "error": repr(error)})
    for branch in ("cache_oof", "linear_oof", "cache_loss_tiebreak"):
        for name in DATASETS:
            directory = raw / branch / name
            source = directory / "summary.json"
            status_path = directory / "status.json"
            if status_path.exists():
                status = read(status_path)
                job_states.append({"branch": branch, "dataset": name, **status})
            if not source.exists():
                continue
            try:
                # Summary is written after each finished candidate, even when a
                # later candidate fails. Keep those completed measurements.
                results = read(source)["results"]
                for rank, result in enumerate(results):
                    config = result["config"]
                    feature = config["feature"]
                    method = config.get("method", "linear")
                    add(f"{branch}:{method}:{feature}", name, result["metrics"],
                        result["baseline_metrics"], source, config,
                        "hyperparameters selected only by grouped support OOF")
                    if rank == 0:
                        add(f"{branch}:support_selected", name, result["metrics"],
                            result["baseline_metrics"], source, config,
                            "method, feature and hyperparameters selected only by grouped support OOF")
            except Exception as error:
                errors.append({"source": str(source), "error": repr(error)})
    for controller in ("controller", "cpu_controller", "tiebreak_controller"):
        source = campaign / controller / "state.json"
        if source.exists():
            try:
                state = read(source)
                jobs = state.get("jobs", {})
                entries = jobs.items() if isinstance(jobs, dict) else (
                    (str(index), job) for index, job in enumerate(jobs))
                for job_id, job in entries:
                    job_states.append({
                        "controller": controller, "job_id": job_id,
                        "status": job.get("status"), "returncode": job.get("returncode"),
                        "error": job.get("error", job.get("reason")),
                        "log": job.get("log"), "source": str(source),
                    })
            except Exception as error:
                errors.append({"source": str(source), "error": repr(error)})
    for name in DATASETS:
        source = raw / "query_refinement" / name / "summary.json"
        if source.exists():
            try:
                result = read(source)
                if result["status"] == "complete":
                    for trial in result["trials"]:
                        add(f"query_refinement:{trial['eta']:g}", name, trial["metrics"],
                            result["baseline_metrics"], source, {"eta": trial["eta"]},
                            "prespecified grid; exploratory test comparison")
            except Exception as error:
                errors.append({"source": str(source), "error": repr(error)})
    for name in DATASETS:
        directory = raw / "i2cl_variants" / name
        source = directory / "summary.json"
        state_path = directory / "status.json"
        if state_path.exists():
            job_states.append({"branch": "i2cl_variants", "dataset": name, **read(state_path)})
        if not source.exists():
            continue
        try:
            result = read(source)
            for trial in result["trials"]:
                add(f"i2cl_variants:{trial['name']}", name, trial["metrics"],
                    trial["baseline_metrics"], source, trial["config"],
                    "prespecified grid; exploratory test comparison")
        except Exception as error:
            errors.append({"source": str(source), "error": repr(error)})
    families = {}
    for row in rows:
        group = families.setdefault(row["family"], {})
        if row["dataset"] in group:
            raise ValueError(f"Duplicate family/dataset: {row['family']}/{row['dataset']}")
        group[row["dataset"]] = row
    ranking = []
    for family, group in families.items():
        ranking.append({
            "family": family, "completed_datasets": len(group),
            "datasets_both_above": sum(x["both_above"] for x in group.values()),
            "all_five_both_above": len(group) == 5 and all(x["both_above"] for x in group.values()),
            "mean_delta_auroc_pp": sum(x["delta_auroc_pp"] for x in group.values()) / len(group),
            "mean_delta_auprc_pp": sum(x["delta_auprc_pp"] for x in group.values()) / len(group),
        })
    ranking.sort(key=lambda x: (-x["all_five_both_above"], -x["completed_datasets"],
                               -x["datasets_both_above"], -x["mean_delta_auroc_pp"],
                               -x["mean_delta_auprc_pp"], x["family"]))
    return rows, ranking, errors, job_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=PROJECT / "runs/overnight_20261020_2150")
    parser.add_argument("--raw", type=Path, default=PROJECT / "local_assets/outputs/overnight_20261020_2150")
    args = parser.parse_args()
    rows, ranking, errors, job_states = collect(args.campaign.resolve(), args.raw.resolve())
    generated = datetime.now(timezone.utc).astimezone().isoformat()
    limitation = (
        "Repeated test-guided method development is exploratory. All test comparisons are "
        "retained; a successful candidate still needs independent evaluation. "
        "Original RadZero is zero-shot; these methods use labeled supports. "
        "AUPRC is trapezoidal PR area, not average precision. Single support seed."
    )
    write(args.campaign / "leaderboard.json", {
        "generated_at": generated, "status": "exploratory_progress",
        "limitation": limitation, "rows": rows, "ranking": ranking, "errors": errors,
        "job_states": job_states,
    })
    path = args.campaign / "leaderboard.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    lines = [
        f"RadZero overnight exploratory experiments — {generated}",
        "Deadline: 2026-10-21 10:00 Asia/Shanghai; allowed GPUs: 3–7, only idle/leased.",
        limitation, "",
        "Family | datasets complete | datasets improving BOTH AUROC/AUPRC | all five achieved",
    ]
    for record in ranking:
        lines.append(f"{record['family']} | {record['completed_datasets']}/5 | "
                     f"{record['datasets_both_above']}/5 | {record['all_five_both_above']}")
    lines.extend(["", "Job status ledger (includes pending/failed/interrupted work):"])
    for state in job_states:
        identity = state.get("job_id", f"{state.get('branch')}/{state.get('dataset')}")
        lines.append(f"{identity}: {state.get('status')}; returncode={state.get('returncode')}; "
                     f"error={state.get('error', state.get('reason'))}")
    lines.extend(["", "Full candidate results (metrics %, differences percentage points):",
                  "Family | Dataset | AUROC | AUPRC | dAUROC | dAUPRC | both improved"])
    for row in rows:
        lines.append(f"{row['family']} | {row['dataset']} | {100*row['auroc']:.5f} | "
                     f"{100*row['auprc']:.5f} | {row['delta_auroc_pp']:+.5f} | "
                     f"{row['delta_auprc_pp']:+.5f} | {row['both_above']}")
    if errors:
        lines.extend(["", "Read/audit errors:", json.dumps(errors, ensure_ascii=False, indent=2)])
    (args.campaign / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"candidate_rows": len(rows), "families": len(ranking),
                      "all_five_winners": [x["family"] for x in ranking if x["all_five_both_above"]],
                      "errors": errors}, ensure_ascii=False))


if __name__ == "__main__":
    main()
