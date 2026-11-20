"""CPU checks for support-only grouped linear residual adaptation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from overnight_cache_methods import normalized_features, sha256, fingerprint
from overnight_linear_residual import (
    fit_linear, grouped_oof, make_group_folds, predict_linear_residual,
    weighted_ranking_objective, run,
)


class LinearResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.x = normalized_features(np.asarray([
            [1., .2, .1], [1., .3, .2], [.1, 1., .1],
            [.1, 1., .3], [.4, .2, 1.], [.2, .1, 1.],
        ]), 2)
        self.y = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]])
        self.z = np.zeros_like(self.y, dtype=float)
        self.groups = ["a", "a", "b", "b", "c", "d"]

    def test_folds_are_disjoint_complete_deterministic_and_keep_groups(self):
        folds = make_group_folds(self.groups, count=3, seed=42)
        self.assertEqual(folds, make_group_folds(self.groups, count=3, seed=42))
        observed = []
        for fold in folds:
            fit, held = fold["fit_indices"], fold["held_indices"]
            observed.extend(held)
            self.assertFalse(set(self.groups[i] for i in fit)
                             & set(self.groups[i] for i in held))
            self.assertEqual(sorted(fit + held), list(range(6)))
        self.assertEqual(sorted(observed), list(range(6)))

    def test_zero_initialization_restores_baseline_and_mean_uses_fit_subset(self):
        state = fit_linear(self.x[:4], self.y[:4], self.z[:4], epochs=0)
        np.testing.assert_array_equal(state["weight"], 0)
        np.testing.assert_array_equal(state["bias"], 0)
        np.testing.assert_array_equal(predict_linear_residual(state, self.x), 0)
        np.testing.assert_allclose(state["means"][0], self.x[:4].mean(axis=0))
        self.assertGreater(np.abs(state["means"][0] - self.x.mean(axis=0)).max(), .01)

    def test_missing_class_states_produce_zero_residual_and_no_weight_update(self):
        labels = self.y.copy()
        labels[:, 0] = 1
        state = fit_linear(self.x, labels, self.z, epochs=10, ranking=.1)
        self.assertFalse(state["valid"][0])
        np.testing.assert_array_equal(state["weight"][0], 0)
        self.assertEqual(state["bias"][0], 0)
        np.testing.assert_array_equal(predict_linear_residual(state, self.x)[:, 0], 0)

    def test_bce_weights_and_pairwise_loss_match_manual_values(self):
        logits = torch.zeros((3, 2), requires_grad=True)
        labels = torch.tensor([[1., 1.], [0., 1.], [0., 0.]])
        positive_weights = torch.tensor([2., .5])
        valid = torch.ones(2)
        weights = torch.ones((2, 3))
        loss, terms = weighted_ranking_objective(
            logits, labels, positive_weights, valid, weights, .2, .1)
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=positive_weights)
        torch.testing.assert_close(terms["weighted_bce"], expected_bce)
        self.assertAlmostEqual(float(terms["mean_class_weight_l2"]), 3.)
        self.assertAlmostEqual(float(terms["pairwise_ranking"]), np.log(2), places=6)
        self.assertAlmostEqual(float(loss.detach()), float(expected_bce.detach()) + .6 + .1*np.log(2),
                               places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_grouped_oof_matches_manual_training_and_uses_fold_means(self):
        folds = make_group_folds(self.groups, count=3, seed=42)
        result, audit = grouped_oof(self.x, self.y, self.z, folds, epochs=3)
        for fold, entry in zip(folds, audit):
            fit, held = fold["fit_indices"], fold["held_indices"]
            state = fit_linear(self.x[fit], self.y[fit], self.z[fit], epochs=3)
            np.testing.assert_allclose(result[held], predict_linear_residual(state, self.x[held]))
            self.assertEqual(entry["fit_positive_counts"], self.y[fit].sum(axis=0).tolist())

    def test_class_specific_and_global_equivalence_and_no_bias(self):
        local = np.repeat(self.x[:, None], 2, axis=1)
        state = fit_linear(self.x, self.y, self.z, epochs=10, use_bias=False)
        local_state = fit_linear(local, self.y, self.z, epochs=10, use_bias=False)
        np.testing.assert_allclose(predict_linear_residual(state, self.x),
                                   predict_linear_residual(local_state, local), atol=1e-7)
        np.testing.assert_array_equal(state["bias"], 0)
        self.assertLess(state["history"][-1]["total_loss"], state["history"][0]["total_loss"])

    def test_pipeline_consumes_schema_and_fixes_selection_before_test_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, split_dir, output = root / "cache", root / "splits", root / "out"
            cache.mkdir()
            (split_dir / "Toy").mkdir(parents=True)
            specs, identities, arrays = {}, {}, {}
            for split in ("support", "test"):
                prefix = "s" if split == "support" else "t"
                (split_dir / "Toy" / f"{split}.csv").write_text(f"{split} fixture")
                metadata = {
                    "name": "Toy", "data_root": str(root),
                    "image_paths": [f"{prefix}{i}.png" for i in range(6)],
                    "ids": [f"{prefix}{i}" for i in range(6)],
                    "groups": self.groups if split == "support" else [f"t{i}" for i in range(6)],
                    "class_names": ["a", "b"], "prompts": ["finding a", "finding b"],
                }
                specs[split] = {**metadata, "labels": self.y}
                identities[split] = fingerprint(metadata)
                (cache / f"{split}_metadata.json").write_text(json.dumps(metadata))
                arrays[split] = {"files": {}}
                for key, values in {
                    "global_dino": np.stack([self.x, self.x], axis=1).astype(np.float32),
                    "global_aligned": np.stack([self.x, self.x], axis=1).astype(np.float32),
                    "class_local": np.repeat(self.x[:, None], 2, axis=1).astype(np.float32),
                    "base_logits": self.z.astype(np.float32),
                }.items():
                    path = cache / f"{split}_{key}.npy"
                    np.save(path, values)
                    arrays[split]["files"][key] = {"file": path.name, "shape": list(values.shape),
                                                 "dtype": str(values.dtype), "sha256": sha256(path)}
            np.save(cache / "support_labels.npy", self.y.astype(np.int8))
            manifest = {
                "status": "complete", "dataset": "Toy", "split_dir": str(split_dir),
                "class_names": ["a", "b"], "splits": arrays,
                "support_identity": identities["support"], "test_identity": identities["test"],
                "split_sha256": {
                    str(split_dir / "Toy" / f"{split}.csv"): sha256(split_dir / "Toy" / f"{split}.csv")
                    for split in ("support", "test")},
                "auxiliary_files": {
                    path.name: {"sha256": sha256(path)}
                    for path in (cache / "support_labels.npy",
                                 cache / "support_metadata.json", cache / "test_metadata.json")},
            }
            (cache / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(
                cache=str(cache), output=str(output), dataset="Toy", folds=3, seed=42,
                features=["class_local", "dino_cls"], l2s=[.1], rankings=[0., .1],
                strengths=[0., .1, 1.], top_k=2, epochs=3, learning_rate=.01,
                use_bias=True, deadline=None, chunk_size=2, threads=1)
            reads = []

            def load_spec(path, name, split):
                self.assertEqual(name, "Toy")
                if split == "test":
                    self.assertTrue((output / "selection.json").exists())
                    selection = json.loads((output / "selection.json").read_text())
                    self.assertTrue(selection["selected_before_test_labels_loaded"])
                    reads.append(split)
                return specs[split]

            with patch("i2cl_data.load_spec", side_effect=load_spec):
                run(args)
            self.assertEqual(len(reads), 2)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["results"]), 2)
            self.assertFalse(summary["test_labels_used_in_fitting_or_selection"])
            for entry in summary["results"]:
                destination = output / entry["candidate_id"]
                self.assertTrue((destination / "linear_residual.npz").is_file())
                self.assertTrue((destination / "predictions.csv").is_file())
            with patch("i2cl_data.load_spec", side_effect=load_spec):
                with self.assertRaises(FileExistsError):
                    run(args)


if __name__ == "__main__":
    unittest.main()
