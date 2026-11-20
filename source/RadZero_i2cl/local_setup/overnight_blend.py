#!/usr/bin/env python3
"""Exploratory, fixed-grid baseline/I2CL score ensembles from saved predictions.

No labels fit this ensemble. Test labels only score the prespecified grid.
Selecting a candidate from these repeated evaluations is explicitly exploratory.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_i2cl import independent_metrics, verify_metrics
from batch_data import DATASETS


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_predictions(path, classes):
    with path.open() as handle:
        records = list(csv.DictReader(handle))
    paths = [row["image_path"] for row in records]
    labels = np.array([[int(row[f"label:{c}"]) for c in classes] for row in records], np.int8)
    logits = np.array([[float(row[f"logit:{c}"]) for c in classes] for row in records], np.float32)
    return paths, labels, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[.03, .1, .3, .5, .7, 1.])
    parser.add_argument("--deadline", default="2026-10-21T10:00:00+08:00")
    args = parser.parse_args()
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s in args.strengths):
        parser.error("strengths must be finite and lie in [0, 1]")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline requires a timezone")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "method": "baseline-anchored I2CL score ensemble",
        "formula": "baseline_logits + strength * (i2cl_logits - baseline_logits)",
        "source": str(args.source.resolve()), "strengths": args.strengths,
        "code_sha256": sha256(__file__), "deadline": args.deadline,
        "classification_only": True,
        "limitation": "Prespecified grid scored repeatedly on test sets; exploratory selection only.",
    }
    write_json(args.output / "protocol.json", protocol)
    prepared = {}
    for name in DATASETS:
        directory = args.source / name
        config = json.loads((directory / "run_config.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        if result["status"] != "complete" or result["is_smoke_test"]:
            raise ValueError("Require completed full-test source: " + name)
        for filename, field in (("predictions.csv", "predictions_sha256"),
                                ("baseline_predictions.csv", "baseline_predictions_sha256")):
            if sha256(directory / filename) != result[field]:
                raise ValueError("Changed source predictions: " + name)
        classes = config["class_names"]
        paths, labels, base = read_predictions(directory / "baseline_predictions.csv", classes)
        paths_new, labels_new, adapted = read_predictions(directory / "predictions.csv", classes)
        if paths_new != paths or not np.array_equal(labels_new, labels):
            raise ValueError("Ensemble members have different evaluation rows")
        baseline = independent_metrics(labels, base, classes)
        verify_metrics(result["baseline_metrics"], baseline, name + "/baseline")
        verify_metrics(result["metrics"], independent_metrics(labels, adapted, classes), name + "/i2cl")
        prepared[name] = (labels, base, adapted, baseline, classes, config)
    trials = []
    for strength in args.strengths:
        if datetime.now(timezone.utc) >= deadline:
            break
        trial = {"strength": strength, "datasets": {}, "all_five_both_metrics_above_baseline": True}
        directory = args.output / f"strength_{strength:g}"
        directory.mkdir()
        for name, (labels, base, adapted, baseline, classes, config) in prepared.items():
            logits = base + np.float32(strength) * (adapted - base)
            if strength == 0:
                assert np.array_equal(logits, base)
            metrics = independent_metrics(labels, logits, classes)
            delta = {key: 100 * (metrics[key] - baseline[key])
                     for key in ("macro_auroc", "macro_auprc")}
            passed = all(value > 0 for value in delta.values())
            trial["all_five_both_metrics_above_baseline"] &= passed
            path = directory / f"{name}_logits.npy"
            np.save(path, logits)
            trial["datasets"][name] = {
                "metrics": metrics, "baseline_metrics": baseline, "delta_pp": delta,
                "both_metrics_above_baseline": passed,
                "support_identity": config["support_identity"], "test_identity": config["test_identity"],
                "logits": str(path.resolve()), "logits_sha256": sha256(path),
            }
        trials.append(trial)
        write_json(directory / "result.json", trial)
        print(json.dumps({"strength": strength,
                          "delta_pp": {k: v["delta_pp"] for k, v in trial["datasets"].items()},
                          "all_five_pass": trial["all_five_both_metrics_above_baseline"]}), flush=True)
    write_json(args.output / "summary.json", {
        "status": "complete" if len(trials) == len(args.strengths) else "deadline_reached",
        "protocol": protocol, "trials": trials,
    })


if __name__ == "__main__":
    main()
