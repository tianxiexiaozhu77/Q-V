#!/usr/bin/env python3
"""Thin CPU supervisor for a wider support-OOF ridge grid; old runs stay frozen."""

import argparse
from pathlib import Path
import signal

from overnight_cpu_control import CPUController, PROJECT, parse_args as base_parse_args, sha256

DEFAULT_CACHE_ROOT = PROJECT / "local_assets/outputs/overnight_20261020_2150/features"
PENALTIES = (.01, .1, 1., 10., 100.)
STRENGTHS = (0., .3, 1., 3., 10., 30.)
FEATURES = ("class_local", "dino_cls", "aligned_cls")


class WideRidgeController(CPUController):
    def __init__(self, args):
        super().__init__(args)
        worker = PROJECT / "local_setup/overnight_oof_loss_tiebreak.py"
        jobs = {}
        for dataset in args.datasets:
            identifier = "ridge_wide_" + dataset
            jobs[identifier] = {
                "id": identifier, "dataset": dataset, "method": "ridge_wide",
                "worker": str(worker), "worker_sha256": sha256(worker),
                "cache": str(args.cache_root / dataset),
                "output": str(args.raw_root / "ridge_wide" / dataset),
                "status": "waiting_for_cache",
            }
        self.state["jobs"] = jobs
        self.state["grid"] = {
            "methods": ["ridge"], "features": list(FEATURES),
            "l2s": list(PENALTIES), "strengths": list(STRENGTHS),
            "candidates_per_dataset": len(FEATURES) * len(PENALTIES) * len(STRENGTHS),
            "selection": "Support-group OOF mean AUROC/AUPRC; exact ties use fold-weighted BCE",
            "test_labels_used_for_selection": False,
        }

    def command_for_job(self, job):
        return [
            "bash", str(PROJECT / "local_setup/python_local.sh"), job["worker"],
            "--cache", job["cache"], "--output", job["output"],
            "--dataset", job["dataset"], "--threads", "4",
            "--deadline", str(self.args.deadline),
            "--methods", "ridge", "--features", *FEATURES,
            "--l2s", *[str(value) for value in PENALTIES],
            "--strengths", *[str(value) for value in STRENGTHS], "--top-k", "3",
        ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    extra, remaining = parser.parse_known_args(argv)
    args = base_parse_args(remaining)
    args.cache_root = extra.cache_root.expanduser().resolve()
    args.stop_files.append(args.state_dir.parent / "STOP")
    return args


def main():
    controller = WideRidgeController(parse_args())
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
