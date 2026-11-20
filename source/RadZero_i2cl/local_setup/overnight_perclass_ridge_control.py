#!/usr/bin/env python3
"""Independent supervisor for per-class OOF ridge; uses proven CPU lifecycle."""

import signal

from overnight_cpu_control import CPUController, PROJECT, sha256
from overnight_ridge_wide_control import parse_args


class PerClassController(CPUController):
    def __init__(self, args):
        super().__init__(args)
        worker = PROJECT / "local_setup/overnight_perclass_ridge.py"
        self.state["jobs"] = {
            "ridge_perclass_" + dataset: {
                "id": "ridge_perclass_" + dataset, "dataset": dataset,
                "method": "ridge_perclass", "worker": str(worker),
                "worker_sha256": sha256(worker),
                "cache": str(args.cache_root / dataset),
                "output": str(args.raw_root / "ridge_perclass" / dataset),
                "status": "waiting_for_cache",
            } for dataset in args.datasets}

    def command_for_job(self, job):
        return [
            "bash", str(PROJECT / "local_setup/python_local.sh"), job["worker"],
            "--cache", job["cache"], "--output", job["output"], "--dataset", job["dataset"],
            "--threads", "4", "--deadline", str(self.args.deadline),
            "--binary-reference-root", str(self.args.raw_root / "ridge_wide"),
        ]


def main():
    controller = PerClassController(parse_args())
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
