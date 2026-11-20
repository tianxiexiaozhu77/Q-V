"""CPU-only validation of the new grid and its existing worker CLI."""

from pathlib import Path
import unittest

from overnight_oof_loss_tiebreak import parse_args as worker_parse_args
from overnight_ridge_wide_control import WideRidgeController, parse_args


class WideRidgeControlTests(unittest.TestCase):
    def make_controller(self):
        args = parse_args([
            "--raw-root", "/tmp/ridge_wide_test_raw",
            "--state-dir", "/tmp/ridge_wide_test_campaign/cpu",
            "--gpu-state", "/tmp/ridge_wide_test_campaign/controller/state.json",
            "--cache-root", "/tmp/old_frozen_features",
        ])
        return WideRidgeController(args)

    def test_all_five_jobs_use_existing_cache_and_fresh_output(self):
        controller = self.make_controller()
        self.assertEqual(len(controller.state["jobs"]), 5)
        self.assertEqual(controller.args.max_parallel, 2)
        for job in controller.state["jobs"].values():
            self.assertEqual(Path(job["cache"]).parent, Path("/tmp/old_frozen_features"))
            self.assertEqual(Path(job["output"]).parent, Path("/tmp/ridge_wide_test_raw/ridge_wide"))
            self.assertEqual(job["status"], "waiting_for_cache")

    def test_generated_command_matches_frozen_worker_cli_and_grid(self):
        controller = self.make_controller()
        job = next(iter(controller.state["jobs"].values()))
        args = worker_parse_args(controller.command_for_job(job)[3:])
        self.assertEqual(args.methods, ["ridge"])
        self.assertEqual(args.features, ["class_local", "dino_cls", "aligned_cls"])
        self.assertEqual(args.l2s, [.01, .1, 1., 10., 100.])
        self.assertEqual(args.strengths, [0., .3, 1., 3., 10., 30.])
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.threads, 4)
        self.assertEqual(args.deadline, controller.args.deadline)
        self.assertEqual(controller.state["grid"]["candidates_per_dataset"], 90)

    def test_stop_links_include_global_and_controller_stops(self):
        stops = self.make_controller().args.stop_files
        self.assertIn(Path("/tmp/ridge_wide_test_campaign/STOP"), stops)
        self.assertIn(Path("/tmp/ridge_wide_test_campaign/controller/STOP"), stops)
        self.assertIn(Path("/tmp/ridge_wide_test_campaign/cpu/STOP"), stops)


if __name__ == "__main__":
    unittest.main()
