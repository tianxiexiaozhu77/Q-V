"""CPU checks of group exclusions, class axes, and support-only score fitting."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from overnight_cache_methods import (
    DeadlineReached, check_deadline, fit_residual, leave_group_out,
    normalized_features, predict_residual, choose_candidates, run, sha256, fingerprint,
)


class CacheMethodTests(unittest.TestCase):
    def setUp(self):
        self.x = normalized_features(np.asarray([
            [1., .2, .1], [1., .3, .2], [.1, 1., .1],
            [.1, 1., .3], [.4, .2, 1.], [.2, .1, 1.],
        ]), 2)
        self.y = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
        self.z = np.zeros_like(self.y, dtype=float)
        self.groups = ["a", "a", "b", "b", "c", "d"]

    def test_group_oof_excludes_every_related_image_and_matches_manual_refit(self):
        for method in ("cache", "proto", "ridge"):
            with self.subTest(method=method):
                actual, audit = leave_group_out(self.x, self.y, self.z, self.groups, method, 1.)
                self.assertEqual(audit["fold_count"], 4)
                for fold in audit["folds"]:
                    train, held = fold["fit_indices"], fold["held_indices"]
                    self.assertFalse(set(self.groups[i] for i in train)
                                     & set(self.groups[i] for i in held))
                    state = fit_residual(self.x[train], self.y[train], self.z[train], method, 1.)
                    np.testing.assert_allclose(actual[held], predict_residual(state, self.x[held]))

    def test_missing_positive_class_in_fold_has_zero_residual(self):
        labels = self.y.copy()
        labels[:, 0] = [1, 1, 0, 0, 0, 0]
        for method in ("cache", "proto", "ridge"):
            actual, audit = leave_group_out(self.x, labels, self.z, self.groups, method, 1.)
            np.testing.assert_array_equal(actual[:2, 0], 0.)
            self.assertEqual(audit["missing_state_rows_per_class"][0], 2)

    def test_class_local_and_repeated_global_features_agree(self):
        local = np.repeat(self.x[:, None, :], 2, axis=1)
        for method in ("cache", "proto", "ridge"):
            state_global = fit_residual(self.x, self.y, self.z, method, 1.)
            state_local = fit_residual(local, self.y, self.z, method, 1.)
            np.testing.assert_allclose(
                predict_residual(state_global, self.x),
                predict_residual(state_local, local), atol=1e-12)

    def test_ridge_dual_equals_explicit_weighted_primal_solution(self):
        state = fit_residual(self.x, self.y, self.z, "ridge", .7)
        for c in range(2):
            positive = self.y[:, c] == 1
            weights = np.where(positive, (~positive).sum() / positive.sum(), 1.)
            weights /= weights.mean()
            expected = np.linalg.solve(
                self.x.T @ (weights[:, None] * self.x) + .7 * np.eye(self.x.shape[1]),
                self.x.T @ (weights * (self.y[:, c] - .5)))
            np.testing.assert_allclose(state["directions"][c], expected, atol=1e-12)

    def test_cache_uses_positive_and_negative_means_not_unnormalized_sums(self):
        state = fit_residual(self.x, self.y, self.z, "cache", 5.)
        actual = predict_residual(state, self.x)
        kernel = np.exp(-5 * (1 - np.clip(self.x @ self.x.T, -1, 1)))
        for c in range(2):
            positive = self.y[:, c] == 1
            expected = kernel[:, positive].mean(axis=1) - kernel[:, ~positive].mean(axis=1)
            np.testing.assert_allclose(actual[:, c], expected)

    def test_no_class_leakage_between_class_conditioned_features(self):
        local = np.stack([self.x, self.x[:, ::-1]], axis=1)
        state = fit_residual(local, self.y, self.z, "proto", 1.)
        query = local.copy()
        query[:, 1] *= -1
        first = predict_residual(state, local)
        second = predict_residual(state, query)
        np.testing.assert_array_equal(first[:, 0], second[:, 0])
        np.testing.assert_allclose(first[:, 1], -second[:, 1])

    def test_selection_uses_only_supplied_oof_score_and_different_pairs(self):
        records = []
        for feature, strength, score in [("a", 1, .6), ("a", 2, .7), ("b", 0, .65)]:
            records.append({"candidate_id": f"{feature}{strength}",
                            "config": {"method": "proto", "feature": feature, "strength": strength},
                            "selection_score": score})
        self.assertEqual([x["candidate_id"] for x in choose_candidates(records, 2)], ["a2", "b0"])

    def test_deadline_and_invalid_features(self):
        with patch("overnight_cache_methods.time.time", return_value=2):
            with self.assertRaises(DeadlineReached):
                check_deadline(1)
        with self.assertRaises(ValueError):
            normalized_features(np.zeros((2, 3)), 1)
        with self.assertRaises(ValueError):
            normalized_features(np.ones((2, 3, 4)), 2)

    def test_cli_pipeline_matches_extractor_schema_and_defers_test_label_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, splits, output = root / "cache", root / "splits", root / "run"
            cache.mkdir()
            (splits / "Toy").mkdir(parents=True)
            specs, identity, arrays = {}, {}, {}
            for split in ("support", "test"):
                prefix = "s" if split == "support" else "t"
                (splits / "Toy" / f"{split}.csv").write_text(f"{split} fixture")
                metadata = {
                    "name": "Toy", "data_root": str(root),
                    "image_paths": [f"{prefix}{i}.png" for i in range(6)],
                    "ids": [f"{prefix}{i}" for i in range(6)],
                    "groups": self.groups if split == "support" else [f"t{i}" for i in range(6)],
                    "class_names": ["class_a", "class_b"], "prompts": ["a", "b"],
                }
                specs[split] = {**metadata, "labels": self.y.copy()}
                identity[split] = fingerprint(metadata)
                (cache / f"{split}_metadata.json").write_text(json.dumps(metadata))
                arrays[split] = {"files": {}}
                values = {
                    "global_dino": np.stack([self.x, self.x], axis=1).astype(np.float32),
                    "global_aligned": np.stack([self.x, self.x], axis=1).astype(np.float32),
                    "class_local": np.repeat(self.x[:, None], 2, axis=1).astype(np.float32),
                    "base_logits": self.z.astype(np.float32),
                }
                for key, value in values.items():
                    path = cache / f"{split}_{key}.npy"
                    np.save(path, value)
                    arrays[split]["files"][key] = {
                        "file": path.name, "sha256": sha256(path),
                        "shape": list(value.shape), "dtype": str(value.dtype),
                    }
            np.save(cache / "support_labels.npy", self.y.astype(np.int8))
            manifest = {
                "status": "complete", "dataset": "Toy", "split_dir": str(splits),
                "splits": arrays, "class_names": ["class_a", "class_b"],
                "support_identity": identity["support"], "test_identity": identity["test"],
                "split_sha256": {
                    str(splits / "Toy" / f"{split}.csv"): sha256(splits / "Toy" / f"{split}.csv")
                    for split in ("support", "test")},
                "auxiliary_files": {
                    path.name: {"sha256": sha256(path)}
                    for path in [cache / "support_labels.npy",
                                 cache / "support_metadata.json", cache / "test_metadata.json"]},
            }
            (cache / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(cache=str(cache), output=str(output), dataset="Toy",
                                   features=["class_local", "dino_cls"], methods=["proto"],
                                   taus=[1.], l2s=[1.], strengths=[0., .3, 1.],
                                   deadline=None, top_k=2, chunk_size=2, threads=1)
            test_reads = []

            def verified_load(directory, name, split):
                self.assertEqual(name, "Toy")
                if split == "test":
                    self.assertTrue((output / "selection.json").exists())
                    selection = json.loads((output / "selection.json").read_text())
                    self.assertTrue(selection["selected_before_test_labels_loaded"])
                    test_reads.append(split)
                return specs[split]

            with patch("i2cl_data.load_spec", side_effect=verified_load):
                run(args)
            self.assertEqual(len(test_reads), 2)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 2)
            self.assertFalse(summary["test_labels_used_in_fitting_or_selection"])
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
            for result in summary["results"]:
                destination = output / result["candidate_id"]
                self.assertTrue((destination / "fit.npz").is_file())
                self.assertTrue((destination / "predictions.csv").is_file())
                self.assertEqual(np.load(destination / "predictions.npy").shape, (6, 2))
            with patch("i2cl_data.load_spec", side_effect=verified_load):
                with self.assertRaises(FileExistsError):
                    run(args)


if __name__ == "__main__":
    unittest.main()
