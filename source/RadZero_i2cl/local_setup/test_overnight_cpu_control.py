"""CPU-only integration checks with tiny temporary stand-in workers."""

import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from overnight_cpu_control import CPUController, atomic_json, parse_args, sha256
from overnight_control import process_identity


class CPUControlTests(unittest.TestCase):
    def setup_controller(self, root, worker_code, seconds=1.5, max_parallel=2):
        args = SimpleNamespace(
            raw_root=root / "raw", state_dir=root / "state", gpu_state=root / "gpu/state.json",
            stop_files=[root / "STOP", root / "gpu/STOP"],
            datasets=["SIIM"], max_parallel=max_parallel,
            deadline=time.time() + seconds, poll_seconds=.02, grace_seconds=.1)
        controller = CPUController(args)
        worker = root / "worker.py"
        worker.write_text(worker_code)
        for job in controller.state["jobs"].values():
            job.update(worker=str(worker), worker_sha256=sha256(worker))
        return controller

    def ready(self, controller, status="complete"):
        path = controller.args.raw_root / "features/SIIM"
        path.mkdir(parents=True)
        atomic_json(path / "manifest.json", {"status": status})

    @staticmethod
    def good_worker():
        return """import argparse,json,os,pathlib,time
p=argparse.ArgumentParser()
p.add_argument('--output')
a,_=p.parse_known_args()
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert all(os.environ[k]=='4' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'])
out=pathlib.Path(a.output)
out.mkdir()
(out/'status.json').write_text(json.dumps({'status':'complete'}))
"""

    def test_cache_ready_runs_both_cpu_workers_and_checks_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.setup_controller(Path(temporary), self.good_worker(), seconds=4)
            self.ready(controller)
            state = controller.run()
            self.assertEqual(state["status"], "complete")
            self.assertTrue(all(job["status"] == "complete" for job in state["jobs"].values()))
            cache_job = state["jobs"]["cache_oof_SIIM"]
            self.assertEqual(cache_job["argv"][-2:], ["--top-k", "9"])

    def test_cache_failure_skips_without_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.setup_controller(Path(temporary), self.good_worker())
            self.ready(controller, status="failed")
            state = controller.run()
            self.assertEqual(state["status"], "finished_with_failures")
            self.assertTrue(all(job["status"] == "skipped" for job in state["jobs"].values()))
            self.assertTrue(all("pid" not in job for job in state["jobs"].values()))

    def test_failed_gpu_producer_skips_missing_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.setup_controller(Path(temporary), self.good_worker())
            controller.args.gpu_state.parent.mkdir()
            atomic_json(controller.args.gpu_state, {
                "jobs": {"features_SIIM": {"status": "failed"}}})
            state = controller.run()
            self.assertTrue(all(job["status"] == "skipped" for job in state["jobs"].values()))

    def test_waiting_cache_respects_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.setup_controller(Path(temporary), self.good_worker(), seconds=.1)
            state = controller.run()
            self.assertEqual(state["stop_reason"], "deadline")
            self.assertTrue(all(job["status"] == "not_started" for job in state["jobs"].values()))

    def test_linked_gpu_stop_terminates_owned_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self.setup_controller(
                root, "import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)",
                seconds=4, max_parallel=1)
            self.ready(controller)
            (root / "gpu").mkdir()

            def stop():
                time.sleep(.4)
                (root / "gpu/STOP").touch()
            writer = threading.Thread(target=stop)
            writer.start()
            state = controller.run()
            writer.join()
            self.assertEqual(state["stop_reason"], "stop_file")
            job = state["jobs"]["cache_oof_SIIM"]
            self.assertEqual(job["status"], "stopped")
            self.assertIsNone(process_identity(job["pid"]))
            self.assertEqual(state["jobs"]["linear_oof_SIIM"]["status"], "not_started")

    def test_zero_exit_without_status_is_failed(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = self.setup_controller(Path(temporary), "pass", seconds=3)
            self.ready(controller)
            state = controller.run()
            self.assertTrue(all(job["status"] == "failed" for job in state["jobs"].values()))

    def test_rejects_late_deadline(self):
        with self.assertRaises(SystemExit):
            parse_args(["--raw-root", "/tmp/raw", "--state-dir", "/tmp/state",
                        "--gpu-state", "/tmp/gpu", "--deadline", "2026-10-21T11:00:00+08:00"])


if __name__ == "__main__":
    unittest.main()
