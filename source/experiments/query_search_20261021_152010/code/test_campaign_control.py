"""CPU-only scheduler tests; no NVIDIA calls or shared GPU leases."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from campaign_control import (
    Controller, TOKEN_KEY, atomic_json, parse_args, process_identity,
    signal_owned_group, validate_job,
)


class FakeLease:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ControllerTests(unittest.TestCase):
    def make_controller(self, root, seconds=0.7, reserve=None):
        args = SimpleNamespace(
            queue=root / "queue.json", state_dir=root / "controller",
            stop_file=root / "STOP", deadline=time.time() + seconds + 0.1,
            gpus=[3, 4], poll_seconds=0.02, grace_seconds=0.1)
        leases = []

        def fake_reserve(gpu):
            self.assertIn(gpu, [3, 4])
            lease = FakeLease()
            leases.append(lease)
            return lease, {"uuid": "GPU-FAKE-" + str(gpu), "physical_index": gpu,
                           "name": "CPU_TEST", "memory_used_before_mib": 0}
        return Controller(args, reserve or fake_reserve), leases

    def job(self, root, name, code):
        return {"id": name, "dataset": "CPU_TEST",
                "argv": [sys.executable, "-c", code],
                "output": str(root / name), "command_marker": code}

    def test_reject_gpu_two(self):
        with self.assertRaises(SystemExit):
            parse_args(["--queue", "/tmp/q", "--state-dir", "/tmp/s", "--gpus", "2"])

    def test_reject_bad_job(self):
        with self.assertRaises(ValueError):
            validate_job({"id": "../bad"})

    def test_empty_queue_exits_at_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller, leases = self.make_controller(Path(tmp), seconds=0.1)
            started = time.monotonic()
            state = controller.run()
            self.assertEqual(state["stop_reason"], "deadline")
            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(leases, [])

    def test_failure_then_appended_job_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller, leases = self.make_controller(root, seconds=2)
            failed = self.job(root, "fail", "raise SystemExit(7)")
            good = self.job(root, "good", "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); assert all(os.environ[key]=='4' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])")
            atomic_json(controller.args.queue, {"jobs": [failed]})

            def append():
                time.sleep(0.2)
                atomic_json(controller.args.queue, {"jobs": [failed, good]})
                time.sleep(0.3)
                controller.args.stop_file.touch()
            writer = threading.Thread(target=append)
            writer.start()
            state = controller.run()
            writer.join()
            self.assertEqual(state["jobs"]["fail"]["status"], "failed")
            self.assertEqual(state["jobs"]["good"]["status"], "complete")
            self.assertEqual(state["jobs"]["good"]["gpu_info"]["physical_index"], 3)
            self.assertEqual(state["stop_reason"], "stop_file")
            self.assertIn("GPU-FAKE-", (root / "controller/logs/good.log").read_text())
            self.assertFalse((root / "good").exists())  # Worker owns output creation.
            self.assertTrue(all(lease.closed for lease in leases))

    def test_deadline_kills_own_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller, leases = self.make_controller(root, seconds=0.2)
            job = self.job(root, "slow", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")
            atomic_json(controller.args.queue, {"jobs": [job]})
            started = time.monotonic()
            state = controller.run()
            self.assertLess(time.monotonic() - started, 2)
            self.assertEqual(state["jobs"]["slow"]["status"], "stopped")
            self.assertIsNone(process_identity(state["jobs"]["slow"]["pid"]))
            self.assertTrue(all(lease.closed for lease in leases))

    def test_busy_lease_does_not_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller, _ = self.make_controller(
                root, seconds=0.1,
                reserve=lambda gpu: (None, {"reason": "shared_lease_busy"}))
            atomic_json(controller.args.queue, {"jobs": [self.job(root, "wait", "pass")]})
            state = controller.run()
            self.assertEqual(state["jobs"]["wait"]["status"], "not_started")

    def test_watchdog_stops_job_during_slow_gpu_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leases = []

            def reserve(gpu):
                if gpu == 4:
                    time.sleep(0.7)
                    return None, {"reason": "gpu_in_use"}
                lease = FakeLease()
                leases.append(lease)
                return lease, {"uuid": "GPU-FAKE-3"}

            controller, _ = self.make_controller(root, seconds=0.15, reserve=reserve)
            slow = self.job(root, "slow", "import time; time.sleep(30)")
            waiting = self.job(root, "wait", "pass")
            atomic_json(controller.args.queue, {"jobs": [slow, waiting]})
            state = controller.run()
            self.assertEqual(state["jobs"]["slow"]["status"], "stopped")
            self.assertEqual(state["jobs"]["wait"]["status"], "not_started")
            self.assertEqual(state["jobs"]["slow"]["returncode"], -signal.SIGTERM)
            self.assertTrue(all(lease.closed for lease in leases))

    def test_identity_mismatch_refuses_signal(self):
        code = "import time; time.sleep(30)"
        env = os.environ.copy()
        env[TOKEN_KEY] = "real-token"
        process = subprocess.Popen([sys.executable, "-c", code], env=env,
                                   start_new_session=True)
        try:
            time.sleep(0.05)
            identity = process_identity(process.pid)
            record = {"pid": process.pid, "start_ticks": identity["start_ticks"],
                      "token": "wrong-token", "command_marker": code}
            self.assertFalse(signal_owned_group(record, signal.SIGTERM))
            self.assertIsNone(process.poll())
            record["token"] = "real-token"
            record["command_marker"] = "wrong-command"
            self.assertFalse(signal_owned_group(record, signal.SIGTERM))
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    unittest.main()
