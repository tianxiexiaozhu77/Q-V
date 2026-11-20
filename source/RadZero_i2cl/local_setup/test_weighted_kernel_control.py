"""CPU command/schema checks; inherited process lifecycle is tested separately."""

from pathlib import Path
import unittest

from overnight_ridge_wide_control import parse_args
from weighted_kernel_control import WeightedKernelController
from weighted_kernel_residual import build_grid, parse_args as worker_args


class WeightedKernelControlTests(unittest.TestCase):
    def controller(self):
        return WeightedKernelController(parse_args([
            "--raw-root", "/tmp/new_kernel_raw", "--state-dir", "/tmp/new_kernel_campaign/cpu",
            "--gpu-state", "/tmp/new_kernel_campaign/controller/state.json"]))

    def test_five_jobs_with_new_output_and_existing_caches(self):
        control = self.controller()
        self.assertEqual(len(control.state["jobs"]), 5)
        self.assertEqual(control.args.max_parallel, 2)
        for job in control.state["jobs"].values():
            self.assertEqual(Path(job["output"]).parent, Path("/tmp/new_kernel_raw/weighted_kernel"))
            self.assertIn("overnight_20261020_2150/features", job["cache"])

    def test_worker_cli_accepts_command_and_preserves_reviewed_grid(self):
        control = self.controller()
        job = next(iter(control.state["jobs"].values()))
        args = worker_args(control.command_for_job(job)[3:])
        self.assertEqual(args.threads, 4)
        self.assertEqual(args.top_k, 4)
        self.assertEqual(args.deadline, control.args.deadline)
        self.assertEqual(len(build_grid(args)) * len(args.strengths), 120)
        self.assertIn(Path("/tmp/new_kernel_campaign/STOP"), control.args.stop_files)


if __name__ == "__main__":
    unittest.main()
