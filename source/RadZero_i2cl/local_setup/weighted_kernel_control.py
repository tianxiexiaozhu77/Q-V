#!/usr/bin/env python3
"""Independent CPU queue for the reviewed weighted-kernel residual experiments."""

import signal

from overnight_cpu_control import CPUController, PROJECT, sha256
from overnight_ridge_wide_control import parse_args


class WeightedKernelController(CPUController):
    def __init__(self, args):
        super().__init__(args)
        worker = PROJECT / "local_setup/weighted_kernel_residual.py"
        self.state["jobs"] = {
            "weighted_kernel_" + dataset: {
                "id": "weighted_kernel_" + dataset, "dataset": dataset,
                "method": "weighted_kernel", "worker": str(worker),
                "worker_sha256": sha256(worker),
                "cache": str(args.cache_root / dataset),
                "output": str(args.raw_root / "weighted_kernel" / dataset),
                "status": "waiting_for_cache",
            } for dataset in args.datasets}
        self.state["selection"] = (
            "Support-group OOF winner per normalize/centered x linear/RBF family; "
            "four candidates retained, one overall support-selected winner marked")

    def command_for_job(self, job):
        return [
            "bash", str(PROJECT / "local_setup/python_local.sh"), job["worker"],
            "--cache", job["cache"], "--output", job["output"], "--dataset", job["dataset"],
            "--threads", "4", "--deadline", str(self.args.deadline), "--top-k", "4",
        ]


def main():
    controller = WeightedKernelController(parse_args())
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
