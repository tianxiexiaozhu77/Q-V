#!/usr/bin/env python3
"""Run the separately recorded OOF-BCE tie-break ablation with the tested CPU supervisor."""
import signal
from overnight_cpu_control import CPUController, PROJECT, parse_args, sha256


def main():
    args = parse_args()
    controller = CPUController(args)
    jobs = {}
    worker = PROJECT / "local_setup/overnight_oof_loss_tiebreak.py"
    for record in controller.state["jobs"].values():
        if record["method"] != "cache_oof":
            continue
        identifier = "cache_loss_tiebreak_" + record["dataset"]
        record.update(
            id=identifier, worker=str(worker), worker_sha256=sha256(worker),
            output=str(args.raw_root / "cache_loss_tiebreak" / record["dataset"]),
            selection_rule="support OOF rank composite, then fold-weighted BCE, then strength",
        )
        jobs[identifier] = record
    controller.state["jobs"] = jobs
    controller.state["experiment"] = "cache_loss_tiebreak"
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: controller.request_stop(
            "signal_" + signal.Signals(number).name))
    controller.run()


if __name__ == "__main__":
    main()
