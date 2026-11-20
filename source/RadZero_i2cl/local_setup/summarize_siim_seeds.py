#!/usr/bin/env python3
"""Summarize all supplied, independently audited SIIM adapter seed runs.

Only the random seed and output directory may differ between run protocols.
Reported dispersion is sample standard deviation (ddof=1), not a confidence
interval or a statistical significance test. No predictions are ensembled and
no run is selected using its test performance.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


INPUT_FILES = (
    "run_manifest.json", "summary.json", "history.json", "audit.json",
    "launcher.json", "progress.json", "comparison.csv",
    "best_adapter.pt", "best_validation_predictions.csv", "test_predictions.csv",
)
SPLITS = ("valid", "test")
METRICS = ("auroc", "auprc")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual, expected, name):
    require(
        isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
        f"{name}: {actual!r} != {expected!r}",
    )


def protocol(manifest):
    result = copy.deepcopy(manifest)
    for key in ("started_at", "fingerprint"):
        result.pop(key)
    result["hyperparameters"].pop("seed")
    result["paths"].pop("output")
    return result


def difference_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.append(field)
            else:
                differences.extend(difference_paths(left[key], right[key], field))
        return differences
    return [] if left == right else [prefix]


def launcher_protocol(command):
    result = list(command)
    for flag in ("--seed", "--output"):
        require(result.count(flag) == 1, f"Launcher must contain exactly one {flag}")
        index = result.index(flag)
        require(index + 1 < len(result), f"Launcher has no value for {flag}")
        result[index + 1] = f"<{flag[2:]}>"
    return result


def launch_value(command, flag):
    require(command.count(flag) == 1, f"Launcher must contain exactly one {flag}")
    index = command.index(flag)
    require(index + 1 < len(command), f"Launcher has no value for {flag}")
    return command[index + 1]


def read_run(path):
    path = Path(path).expanduser().resolve()
    require(path.is_dir(), f"Run directory does not exist: {path}")
    for filename in INPUT_FILES:
        require((path / filename).is_file(), f"Missing completed-run artifact: {path / filename}")
    manifest = read_json(path / "run_manifest.json")
    summary = read_json(path / "summary.json")
    history = read_json(path / "history.json")
    audit = read_json(path / "audit.json")
    launcher = read_json(path / "launcher.json")
    progress = read_json(path / "progress.json")
    seed = manifest["hyperparameters"]["seed"]
    require(isinstance(seed, int) and not isinstance(seed, bool), f"{path}: seed must be an integer")
    require(summary["status"] == progress["status"] == "complete",
            f"Seed {seed}: full experiment is not complete")
    require(launcher["status"] == "complete" and launcher["exit_code"] == 0,
            f"Seed {seed}: launcher did not complete successfully")
    require(audit["status"] == "passed" and audit["errors"] == [],
            f"Seed {seed}: independent audit has not passed")
    require(all(manifest["hyperparameters"][f"limit_{split}"] == 0
                for split in ("train", "valid", "test")),
            f"Seed {seed}: a smoke/subsample run cannot enter the full summary")
    require(Path(manifest["paths"]["output"]).resolve() == path
            and Path(audit["output"]).resolve() == path
            and Path(launcher["output"]).resolve() == path,
            f"Seed {seed}: artifact output paths disagree")
    expected_fingerprint = fingerprint({
        key: value for key, value in manifest.items()
        if key not in ("fingerprint", "started_at")
    })
    require(manifest["fingerprint"] == summary["run_fingerprint"]
            == audit["run_fingerprint"] == expected_fingerprint,
            f"Seed {seed}: run fingerprints disagree")
    require(summary["history"] == history and isinstance(history, list) and bool(history),
            f"Seed {seed}: history differs between files or is empty")
    require(summary["epochs_completed"] == audit["epochs_completed"] == len(history),
            f"Seed {seed}: epoch counts disagree")
    require([entry["epoch"] for entry in history] == list(range(1, len(history) + 1)),
            f"Seed {seed}: history epochs are not consecutive")
    best_epoch = summary["best_epoch"]
    require(best_epoch == audit["best_epoch"] and 1 <= best_epoch <= len(history),
            f"Seed {seed}: selected epoch disagrees with audit")
    require(summary["early_stopped"] == audit["early_stopped"],
            f"Seed {seed}: early-stopping status disagrees")
    require(summary["counts"] == audit["counts"], f"Seed {seed}: dataset counts disagree")
    require(summary["original_model_frozen"] is True
            and summary["frozen_scoring_sha256_before"] == summary["frozen_scoring_sha256_after"]
            == audit["official_and_reported_frozen_scoring_sha256"],
            f"Seed {seed}: frozen-weight verification failed")
    require(manifest["checkpoint"]["weight_sha256"] == audit["official_checkpoint_sha256"],
            f"Seed {seed}: audited official checkpoint differs")
    require(summary["parameters"]["trainable"] == audit["trainable_parameters"],
            f"Seed {seed}: trainable parameter count differs")
    require(summary["baseline_test_equivalence"]["passed"] is True,
            f"Seed {seed}: baseline equivalence failed")
    require(manifest["world_size"] == 2 and manifest["physical_gpus"] == "0,1"
            and launcher["gpus"] == audit["gpu_ids"] == [0, 1]
            and launcher["cuda_visible_devices"] == "0,1",
            f"Seed {seed}: expected physical GPUs 0 and 1")
    command = launcher["command"]
    require(int(launch_value(command, "--seed")) == seed
            and Path(launch_value(command, "--output")).resolve() == path,
            f"Seed {seed}: launcher arguments disagree with run manifest")
    require(len(summary["comparison"]) == len(audit["comparison"]) == 2,
            f"Seed {seed}: expected exactly validation and test comparison rows")
    comparisons = {row["split"]: row for row in summary["comparison"]}
    audited_comparisons = {row["split"]: row for row in audit["comparison"]}
    require(set(comparisons) == set(audited_comparisons) == set(SPLITS),
            f"Seed {seed}: missing validation/test comparison")
    row = {
        "seed": seed, "run_directory": str(path),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "early_stopped": summary["early_stopped"],
    }
    for split in SPLITS:
        comparison = comparisons[split]
        checked = audited_comparisons[split]
        require(comparison["images"] == checked["images"] == summary["counts"][split],
                f"Seed {seed}: {split} sample counts disagree")
        row[f"{split}_images"] = comparison["images"]
        for metric in METRICS:
            baseline = comparison[f"baseline_{metric}"]
            adapter = comparison[f"adapter_{metric}"]
            require(0 <= baseline <= 1 and 0 <= adapter <= 1,
                    f"Seed {seed}: invalid {split} {metric}")
            delta = 100 * (adapter - baseline)
            for key in (f"baseline_{metric}", f"adapter_{metric}",
                        f"delta_{metric}_percentage_points"):
                close(comparison[key], checked[key], f"Seed {seed}: {split} audited {key}")
            close(comparison[f"delta_{metric}_percentage_points"], delta,
                  f"Seed {seed}: {split} {metric} delta")
            if split == "valid":
                close(adapter, history[best_epoch - 1][f"valid_{metric}"],
                      f"Seed {seed}: selected validation {metric}")
            row[f"{split}_baseline_{metric}"] = baseline
            row[f"{split}_adapter_{metric}"] = adapter
            row[f"{split}_delta_{metric}_percentage_points"] = delta
    identity = {
        "manifest_protocol": protocol(manifest),
        "launcher_protocol": launcher_protocol(command),
        "counts": summary["counts"],
        "parameters": summary["parameters"],
        "selection": summary["selection"],
        "cache_fingerprint": summary["cache_fingerprint"],
        "split_fingerprint": audit["split_fingerprint"],
        "frozen_scoring_sha256": summary["frozen_scoring_sha256_after"],
        "selection_min_delta": audit["selection_min_delta"],
    }
    return {
        "row": row, "identity": identity, "run_fingerprint": manifest["fingerprint"],
        "input_sha256": {name: sha256(path / name) for name in INPUT_FILES},
        "audit_passed_at": audit["audited_at"],
    }


def summarize(runs):
    require(len(runs) >= 2, "At least two completed seed runs are required for sample SD")
    seeds = [run["row"]["seed"] for run in runs]
    require(len(set(seeds)) == len(seeds), f"Duplicate random seeds: {seeds}")
    reference = runs[0]
    for run in runs[1:]:
        differences = difference_paths(reference["identity"], run["identity"])
        require(not differences,
                f"Seed {run['row']['seed']} protocol differs from seed "
                f"{reference['row']['seed']}: {', '.join(differences)}")
        for split in SPLITS:
            for metric in METRICS:
                key = f"{split}_baseline_{metric}"
                close(run["row"][key], reference["row"][key],
                      f"Seed {run['row']['seed']}: shared baseline {key}")
    runs = sorted(runs, key=lambda item: item["row"]["seed"])
    aggregate = {}
    for split in SPLITS:
        aggregate[split] = {"images": reference["row"][f"{split}_images"]}
        for metric in METRICS:
            values = [run["row"][f"{split}_adapter_{metric}"] for run in runs]
            deltas = [run["row"][f"{split}_delta_{metric}_percentage_points"] for run in runs]
            baseline = reference["row"][f"{split}_baseline_{metric}"]
            aggregate[split][metric] = {
                "baseline": baseline,
                "adapter_mean": statistics.mean(values),
                "adapter_sample_sd": statistics.stdev(values),
                "adapter_mean_percent": 100 * statistics.mean(values),
                "adapter_sample_sd_percentage_points": 100 * statistics.stdev(values),
                "delta_mean_percentage_points": statistics.mean(deltas),
                "delta_sample_sd_percentage_points": statistics.stdev(deltas),
                "all_seeds_above_baseline": all(value > baseline for value in values),
            }
    return {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Full cleaned SIIM supervised adapter experiments with fixed train/valid/test splits",
        "seed_count": len(runs),
        "seeds": [run["row"]["seed"] for run in runs],
        "statistics": {
            "center": "Arithmetic mean across all supplied random seeds",
            "dispersion": "Sample standard deviation across random seeds",
            "ddof": 1,
            "metric_unit": "AUROC and AUPRC are fractions in [0,1]; *_percent fields multiply by 100",
            "delta_unit": "Percentage points, computed as 100 * (adapter - baseline)",
            "auprc_definition": "Trapezoidal area under the precision-recall curve, not average precision",
            "interpretation": "Describes training-seed variability on one fixed dataset split; "
                              "not a confidence interval or a statistical significance test",
            "selection": "Each seed uses its own validation-selected epoch. All supplied seeds "
                         "are reported; no test-based seed selection or prediction ensembling.",
        },
        "aggregate": aggregate,
        "per_seed": [run["row"] for run in runs],
        "shared_protocol": reference["identity"],
        "shared_protocol_sha256": fingerprint(reference["identity"]),
        "verification": {
            "all_runs_complete": True,
            "all_independent_audits_passed": True,
            "unique_seeds": True,
            "matching_protocol_except_seed_and_output": True,
            "matching_fixed_baseline": True,
            "inputs": [{
                "seed": run["row"]["seed"],
                "run_directory": run["row"]["run_directory"],
                "run_fingerprint": run["run_fingerprint"],
                "audit_passed_at": run["audit_passed_at"],
                "sha256": run["input_sha256"],
            } for run in runs],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path,
                        help="Include every completed seed run; each must have a passed audit.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for per_seed.csv and summary.json")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    inputs = [path.expanduser().resolve() for path in args.runs]
    require(len(set(inputs)) == len(inputs), "The same run directory was supplied more than once")
    require(all(output != path and output not in path.parents and path not in output.parents
                for path in inputs),
            "Summary output must be separate from all input run directories")
    require(not output.exists(), f"Refusing to overwrite an existing output path: {output}")
    result = summarize([read_run(path) for path in inputs])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_seed"][0]))
        writer.writeheader()
        writer.writerows(result["per_seed"])
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"], "seeds": result["seeds"],
        "aggregate": result["aggregate"], "output": str(output),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
