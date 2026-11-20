"""Synthetic CPU integration; no real support fitting, test labels, or GPU calls."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import fit_unit as worker
import run_campaign as launcher
from experiment import METHODS, read_json, sha256, verify_fit_commit, write_json


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        def features(n):
            x = rng.normal(size=(n, 8)).astype(np.float32)
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        labels = np.tile([[0, 1], [1, 0]], (5, 1)).astype(np.int64)
        folds = []
        for i in range(5):
            validation = np.arange(2*i, 2*i+2)
            folds.append(dict(train=np.setdiff1d(np.arange(10), validation), validation=validation))
        manifest = dict(unit_id="toy_data/seed_42", model="toy", dataset="data", seed=42,
                        feature_definition="synthetic unit features")
        self.unit = SimpleNamespace(
            unit_id=manifest["unit_id"], support_features=features(10), labels=labels,
            positive_text_raw=rng.normal(size=(2, 8)).astype(np.float32),
            negative_text_raw=rng.normal(size=(2, 8)).astype(np.float32),
            groups=np.array([f"s{i}" for i in range(10)]), folds=folds,
            test_features=features(6), test_paths=tuple(f"test{i}" for i in range(6)),
            test_groups=tuple(f"group{i}" for i in range(6)), classes=("a", "b"),
            native_predictions={k:rng.normal(size=(6,2)).astype(np.float32)
                                for k in ("original", "query", "view", "query_view")},
            provenance=manifest, manifest=lambda:manifest)
        self.snapshot = dict(config=dict(models=["toy"], datasets=["data"], seeds=[42],
            catalog_path="unused", source_run="unused", text_base_policy="synthetic raw",
            taskres=dict(epoch_candidates=[0, 2], learning_rate_candidates=[0.002],
                         schedule_epochs=2, residual_scale=0.5),
            sstext_plus=dict(text_lambda="adaptive_shots", balanced_importance=False, repel=True)),
            unit_manifests={self.unit.unit_id:manifest}, source_sha256={},
            backbone_scales={"toy":dict(multiplier=10., natural_log=float(np.log(10.)))})
        write_json(self.run/"experiment_snapshot.json", self.snapshot)

    def fit(self):
        with patch.object(worker, "load_unit", return_value=self.unit):
            worker.fit_unit(self.run, "toy", "data", 42)

    def test_worker_roundtrip_resume_and_barrier(self):
        self.fit()
        folder = self.run/self.unit.unit_id
        commit = verify_fit_commit(folder, self.snapshot, sha256(self.run/"experiment_snapshot.json"), self.unit.manifest())
        self.assertFalse(commit["test_labels_opened"])
        digest = sha256(folder/"fit_commit.json")
        with np.load(folder/"predictions.npz", allow_pickle=False) as predictions:
            self.assertEqual(set(predictions.files), {*METHODS, "image_paths", "class_names", "group_ids"})
            for method, expected in self.unit.native_predictions.items():
                np.testing.assert_array_equal(predictions[method], expected)
        with patch.object(worker, "fit_taskres", side_effect=AssertionError("must reuse")):
            self.fit()
        self.assertEqual(digest, sha256(folder/"fit_commit.json"))
        launcher.write_barrier(self.run, self.snapshot)
        self.assertEqual(read_json(self.run/"predictions_complete.json")["status"], "all_predictions_committed")
        barrier_digest = sha256(self.run/"predictions_complete.json")
        launcher.write_barrier(self.run, self.snapshot)
        self.assertEqual(barrier_digest, sha256(self.run/"predictions_complete.json"))

    def test_incomplete_run_cannot_unlock_evaluator(self):
        with self.assertRaises(FileNotFoundError):
            launcher.write_barrier(self.run, self.snapshot)
        self.assertFalse((self.run/"predictions_complete.json").exists())

    def test_corrupt_artifact_is_not_silently_refit(self):
        self.fit()
        (self.run/self.unit.unit_id/"fit_info.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.fit()

    def test_sealed_run_disallows_new_fits(self):
        write_json(self.run/"predictions_complete.json", {})
        with self.assertRaisesRegex(ValueError, "sealed"):
            self.fit()

    def test_gpu_lease_rejects_existing_jobs(self):
        with patch.object(launcher.tempfile, "gettempdir", return_value=self.tmp.name), \
             patch.object(launcher.subprocess, "check_output", side_effect=["GPU-toy\n", "GPU-toy, 100\n", "GPU-toy, 5678\n"]), \
             patch.object(launcher.os, "killpg", side_effect=AssertionError("never kill existing jobs")):
            lease, info = launcher.acquire_gpu(0)
        self.assertIsNone(lease)
        self.assertEqual(info["reason"], "gpu_in_use")

    def test_gpu_uuid_selection_and_release(self):
        with patch.object(launcher.tempfile, "gettempdir", return_value=self.tmp.name), \
             patch.object(launcher.subprocess, "check_output", side_effect=["GPU-toy\n", "GPU-toy, 10\n", ""]):
            lease, info = launcher.acquire_gpu(1)
        self.assertIsNotNone(lease)
        self.assertEqual(launcher.worker_environment(info)["CUDA_VISIBLE_DEVICES"], "GPU-toy")
        launcher.release_gpu(lease)
        self.assertEqual(launcher.worker_environment()["CUDA_VISIBLE_DEVICES"], "")

    def test_exited_child_does_not_abort_cleanup(self):
        child = SimpleNamespace(pid=123, poll=lambda: None)
        with patch.object(launcher.os, "killpg", side_effect=ProcessLookupError):
            launcher.signal_owned_process(child, launcher.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
