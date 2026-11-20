"""CPU tests for GPU ownership and independent experiment failure handling."""

import fcntl
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import run_i2cl as runner


class SchedulerTests(unittest.TestCase):
    def test_regularization_defaults_and_invalid_strength(self):
        args = runner.parser().parse_args([])
        runner.validate(args)
        self.assertEqual(args.lambda_reg, .01)
        self.assertEqual(args.split_dir, (runner.ASSETS / "splits/native_public5_local_seed42").resolve())
        self.assertEqual(args.shots_per_class, 5)
        self.assertEqual(args.learning_rate, .01)
        self.assertEqual(args.epochs, 100)
        for value in (-1., float("nan"), float("inf")):
            args.lambda_reg = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "lambda-reg"):
                runner.validate(args)
        args.lambda_reg = 0.
        runner.validate(args)

    def test_idle_gpu_lease_is_held_until_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner, "GPU_LOCKS", Path(temporary)), patch.object(
                runner.subprocess, "check_output",
                side_effect=["GPU-test, RTX 3090, 15\n", "GPU-other, 42\n"]
            ):
                lease, info = runner.reserve_idle_gpu(2)
                self.assertEqual(info["uuid"], "GPU-test")
                competing = (Path(temporary) / "eval_gpu_2.lock").open("a+")
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lease.close()
                fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                competing.close()

    def test_busy_gpu_is_skipped_and_unlocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(runner, "GPU_LOCKS", Path(temporary)), patch.object(
                runner.subprocess, "check_output",
                side_effect=["GPU-test, RTX 3090, 15\n", "GPU-test, 1234\n"]
            ):
                lease, info = runner.reserve_idle_gpu(4)
                self.assertIsNone(lease)
                self.assertEqual(info["reason"], "gpu_in_use")
                with (Path(temporary) / "eval_gpu_4.lock").open("a+") as competing:
                    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_unauthorized_gpu_is_rejected_before_query(self):
        with patch.object(runner.subprocess, "check_output") as query:
            for gpu in (0, 1, 8):
                with self.assertRaises(ValueError):
                    runner.reserve_idle_gpu(gpu)
            query.assert_not_called()

    def test_preparation_and_worker_failures_do_not_stop_other_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            args = types.SimpleNamespace(
                output=output, split_dir=root / "splits", gpus=[2, 3],
                datasets=["RSNA", "SIIM", "ChestXDet10"], limit_test=0,
                seed=42, shots_per_class=5, epochs=1, grad_batch_size=2,
                eval_batch_size=2, workers=0, learning_rate=.01,
                weight_decay=.001, noise_scale=.001, lambda_reg=.123,
            )

            def prepare(directory, **kwargs):
                if kwargs["datasets"] == ["ChestXDet10"]:
                    raise FileNotFoundError("missing dataset")

            fake_data = types.SimpleNamespace(
                prepare_splits=prepare, load_spec=lambda *args, **kwargs: None)
            launched = []

            def popen(command, **kwargs):
                self.assertEqual(command[command.index("--lambda-reg") + 1], "0.123")
                name = command[command.index("--dataset") + 1]
                directory = Path(command[command.index("--output") + 1])
                launched.append((name, kwargs))
                code = 1 if name == "RSNA" else 0
                if code == 0:
                    (directory / "result.json").write_text(json.dumps({
                        "status": "complete", "baseline_metrics": {"macro_auroc": .7},
                        "metrics": {"macro_auroc": .8}, "delta_pp": {},
                        "support_count": 10, "evaluated_count": 20, "is_smoke_test": False,
                    }))
                return types.SimpleNamespace(pid=1000 + len(launched), poll=lambda: code)

            def reserve(gpu):
                return (root / f"lease_{gpu}").open("a+"), {"uuid": f"GPU-{gpu}"}

            with patch.dict("sys.modules", {"i2cl_data": fake_data}), \
                    patch.object(runner, "reserve_idle_gpu", side_effect=reserve), \
                    patch.object(runner.subprocess, "Popen", side_effect=popen), \
                    patch.object(runner.time, "sleep"), patch.object(runner, "stop") as stop:
                self.assertFalse(runner.run(args))
            stop.assert_not_called()
            self.assertEqual([name for name, _ in launched], ["RSNA", "SIIM"])
            self.assertTrue(all(options["close_fds"] for _, options in launched))
            self.assertEqual([options["env"]["CUDA_VISIBLE_DEVICES"] for _, options in launched],
                             ["GPU-2", "GPU-3"])
            state = json.loads((output / "launcher.json").read_text())
            self.assertEqual(state["status"], "partial_failure")
            self.assertEqual(state["jobs"]["RSNA"]["status"], "failed")
            self.assertEqual(state["jobs"]["SIIM"]["status"], "complete")
            self.assertEqual(state["jobs"]["ChestXDet10"]["status"], "preparation_failed")


if __name__ == "__main__":
    unittest.main()
