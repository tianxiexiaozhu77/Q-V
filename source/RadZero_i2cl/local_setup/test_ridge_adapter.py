"""CPU-only compact adapter tests, with optional local five-set integration."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ridge_adapter import (
    DATASETS, DEFAULT_ADAPTER_ROOT, NPZ_KEYS, RAW, RidgeAdapter, find_selected,
)
from overnight_cache_methods import normalized_features, predict_residual


class RidgeAdapterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.weights = rng.normal(size=(2, 768))
        self.features = rng.normal(size=(7, 2, 768)).astype(np.float32)
        self.base = rng.normal(size=(7, 2)).astype(np.float32)
        self.adapter = RidgeAdapter(self.weights, .3, ["a", "b"], ["There is a", "There is b"],
                                    {"dataset": "Toy", "support_data_in_archive": False,
                                     "test_labels_in_archive": False})

    def test_export_reload_has_only_inference_fields_and_same_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.npz"
            self.adapter.save(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), NPZ_KEYS)
                self.assertEqual(data["directions"].dtype, np.float64)
                self.assertNotIn("support_features", data.files)
                self.assertNotIn("support_labels", data.files)
            loaded = RidgeAdapter.load(path)
            np.testing.assert_array_equal(
                loaded.predict(self.base, self.features), self.adapter.predict(self.base, self.features))
            with self.assertRaises(FileExistsError):
                self.adapter.save(path)

    def test_matches_existing_ridge_raw_score_formula(self):
        state = {
            "class_count": 2, "method": "ridge", "valid": np.array([True, True]),
            "directions": self.weights,
        }
        expected = self.base.astype(np.float64) + .3 * predict_residual(
            state, normalized_features(self.features, 2))
        np.testing.assert_array_equal(self.adapter.predict(self.base, self.features), expected)

    def test_predict_rejects_labels_and_does_not_mutate_inputs(self):
        base, features = self.base.copy(), self.features.copy()
        first = self.adapter.predict(base, features)
        with self.assertRaises(TypeError):
            self.adapter.predict(base, features, labels=np.ones_like(base))
        np.testing.assert_array_equal(base, self.base)
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(first, self.adapter.predict(base, features))

    def test_cli_supports_explicit_adapter_and_defaults_to_five_datasets(self):
        from eval_ridge_adapter import parse_args
        with patch("sys.argv", ["eval_ridge_adapter.py", "--output", "/tmp/new_ridge_eval"]):
            args = parse_args()
            self.assertEqual(tuple(args.datasets), DATASETS)
            self.assertFalse(hasattr(args, "deadline"))
        with patch("sys.argv", ["eval_ridge_adapter.py", "--output", "/tmp/new_ridge_eval",
                                "--adapter", "/tmp/adapter.npz", "--datasets", "SIIM",
                                "--limit-test", "2", "--device", "cpu"]):
            args = parse_args()
            self.assertEqual(args.adapter, Path("/tmp/adapter.npz"))
            self.assertEqual(args.limit_test, 2)

    def test_invalid_dimensions_and_corrupted_weight_payload_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.predict(self.base, self.features[:, :, :767])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.npz"
            self.adapter.save(path)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            data["directions"] = data["directions"] + 1.
            np.savez(path, **data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                RidgeAdapter.load(path)

    def test_all_five_exported_adapters_match_every_cached_test_prediction(self):
        if not (DEFAULT_ADAPTER_ROOT / "manifest.json").exists():
            self.skipTest("Local five-dataset export has not been generated")
        total = 0
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                root = DEFAULT_ADAPTER_ROOT / dataset
                adapter = RidgeAdapter.load(root / "adapter.npz")
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertTrue(manifest["cached_prediction_equivalence"]["bitwise_equal"])
                self.assertFalse(adapter.metadata["support_data_in_archive"])
                self.assertFalse(adapter.metadata["test_labels_in_archive"])
                cache_path = Path(adapter.metadata["cache_manifest"])
                cache = json.loads(cache_path.read_text())
                test_files = cache["splits"]["test"]["files"]
                feature = np.load(cache_path.parent / test_files["class_local"]["file"],
                                  mmap_mode="r", allow_pickle=False)
                base = np.load(cache_path.parent / test_files["base_logits"]["file"],
                               mmap_mode="r", allow_pickle=False)
                run = RAW / "cache_loss_tiebreak" / dataset
                chosen = find_selected(run)
                expected = np.load(run / chosen["candidate_id"] / "predictions.npy",
                                   mmap_mode="r", allow_pickle=False)
                for start in range(0, len(expected), 512):
                    stop = min(start + 512, len(expected))
                    actual = adapter.predict(base[start:stop], feature[start:stop])
                    np.testing.assert_array_equal(actual, expected[start:stop])
                total += len(expected)
        self.assertEqual(total, 492 + 21101 + 7294 + 1694 + 5327)


if __name__ == "__main__":
    unittest.main()
