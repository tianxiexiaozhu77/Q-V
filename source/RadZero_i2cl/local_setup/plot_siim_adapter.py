#!/usr/bin/env python3
"""Plot recorded SIIM adapter training/validation curves; no GPU is required."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Existing SIIM adapter output directory; plots are saved here.")
    args = parser.parse_args()
    directory = args.run_dir.expanduser().resolve()
    manifest = read_json(directory / "run_manifest.json")
    if manifest.get("version") != "frozen-radzero-siim-residual-adapter-v1":
        parser.error("--run-dir must contain an SIIM adapter run")
    history = read_json(directory / "history.json")
    baseline = read_json(directory / "baseline_validation_metrics.json")
    summary_path = directory / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    if not history:
        parser.error("No completed training epoch is available yet")
    epochs = [entry["epoch"] for entry in history]
    if epochs != sorted(set(epochs)):
        parser.error("History epochs must be unique and increasing")
    fields = ("train_weighted_bce", "valid_auroc", "valid_auprc")
    if not all(math.isfinite(entry[field]) for entry in history for field in fields):
        parser.error("Training history contains non-finite metrics")
    for key in ("macro_auroc", "macro_auprc"):
        if not isinstance(baseline.get(key), (int, float)) or not math.isfinite(baseline[key]):
            parser.error("Baseline validation metrics are missing or non-finite")
    # Respect the trainer's min_delta selection, not a newly chosen argmax.
    best_epoch = summary.get("best_epoch", history[-1]["best_epoch"])
    if best_epoch not in epochs:
        best_epoch = history[-1]["best_epoch"]
    if best_epoch not in epochs:
        parser.error("Selected epoch is absent from the saved history")
    finished = (summary.get("status") in ("complete", "smoke_complete")
                and summary.get("epochs_completed") == len(history))
    state = "completed" if finished else "recorded epochs; training may still be running"
    if summary.get("status") == "smoke_complete":
        state = "smoke run completed"

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    panels = (
        ("train_weighted_bce", "Training loss", "Weighted BCE", "#285f9e", None),
        ("valid_auroc", "Validation AUROC", "AUROC", "#16837a", "macro_auroc"),
        ("valid_auprc", "Validation AUPRC", "AUPRC (trapezoidal)", "#c77625", "macro_auprc"),
    )
    for axis, (field, title, ylabel, color, baseline_key) in zip(axes, panels):
        values = [entry[field] for entry in history]
        axis.plot(epochs, values, color=color, marker="o", markersize=4,
                  linewidth=1.8, label="Adapter")
        if baseline_key:
            value = baseline[baseline_key]
            axis.axhline(value, color="#666666", linestyle="--", linewidth=1.2,
                         label=f"Frozen RadZero ({value:.4f})")
        axis.axvline(best_epoch, color="#b44949", linestyle=":", linewidth=1.4,
                     label=f"Selected epoch {best_epoch}")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.set_xlim(min(epochs) - 0.4, max(epochs) + 0.4)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(f"SIIM supervised adapter training — {state}", fontsize=13)
    for suffix in ("png", "pdf"):
        target = directory / ("training_curves." + suffix)
        temporary = target.with_name(target.name + ".tmp")
        figure.savefig(temporary, format=suffix, dpi=180, facecolor="white")
        temporary.replace(target)
        print(target)
    plt.close(figure)


if __name__ == "__main__":
    main()
