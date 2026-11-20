"""CPU checks for weighted kernel algebra, grouped centering and label isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from overnight_cache_methods import (
    DeadlineReached, array_digest, fingerprint, fit_residual, normalized_features,
    predict_residual, sha256, sigmoid,
)
from weighted_kernel_residual import (
    build_grid, choose_candidates, config_key, family_name, fit_kernel, kernel_matrix,
    leave_group_out_grid, load_fit, parse_args, predict_kernel, prepare_fit, run,
    save_fit, transform_features,
)


class KernelResidualTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(41)
        self.x = rng.normal(size=(8, 2, 5))
        self.y = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1],
                             [1, 0], [0, 1], [0, 1], [0, 0]], dtype=np.int8)
        self.z = rng.normal(size=self.y.shape) * .2
        self.groups = ["a", "a", "b", "b", "c", "d", "e", "f"]
        self.config = {"transform": "normalize", "kernel": "linear", "tau": 1., "l2": .7}

    def test_linear_matches_prior_class_balanced_ridge_and_explicit_primal(self):
        state = fit_kernel(self.x, self.y, self.z, self.config)
        normalized = normalized_features(self.x, 2)
        previous = fit_residual(normalized, self.y, self.z, "ridge", .7)
        np.testing.assert_allclose(predict_kernel(state, self.x),
                                   predict_residual(previous, normalized), atol=2e-14)
        for c in range(2):
            positive = self.y[:, c] == 1
            weights = np.where(positive, (~positive).sum() / positive.sum(), 1.)
            weights /= weights.mean()
            x = normalized[:, c]
            expected = np.linalg.solve(
                x.T @ (weights[:, None] * x) + .7 * np.eye(5),
                x.T @ (weights * (self.y[:, c] - sigmoid(self.z[:, c]))))
            np.testing.assert_allclose(state["directions"][c], expected, atol=2e-14)

    def test_rbf_matches_direct_weighted_kernel_equations(self):
        config = {**self.config, "kernel": "rbf", "tau": 5.}
        state = fit_kernel(self.x, self.y, self.z, config)
        query = self.x[:3] + .1
        prediction = predict_kernel(state, query)
        source = normalized_features(self.x, 2)
        test = normalized_features(query, 2)
        for c in range(2):
            positive = self.y[:, c] == 1
            weights = np.where(positive, (~positive).sum() / positive.sum(), 1.)
            weights /= weights.mean()
            k = np.exp(5 * (np.clip(source[:, c] @ source[:, c].T, -1, 1) - 1))
            expected_dual = np.linalg.solve(k + .7 * np.diag(1 / weights),
                                            self.y[:, c] - sigmoid(self.z[:, c]))
            query_kernel = np.exp(5 * (np.clip(test[:, c] @ source[:, c].T, -1, 1) - 1))
            np.testing.assert_allclose(state["dual"][:, c], expected_dual, atol=2e-14)
            np.testing.assert_allclose(prediction[:, c], query_kernel @ expected_dual, atol=2e-14)

    def test_group_oof_and_center_are_refit_after_removing_entire_group(self):
        configs = [self.config, {**self.config, "transform": "support_center"},
                   {**self.config, "transform": "support_center", "kernel": "rbf", "tau": 5.}]
        residuals, audits = leave_group_out_grid(self.x, self.y, self.z, self.groups, configs)
        for config in configs:
            for fold in audits[config["transform"]]["folds"]:
                fit, held = fold["fit_indices"], fold["held_indices"]
                self.assertFalse(set(self.groups[i] for i in fit)
                                 & set(self.groups[i] for i in held))
                state = fit_kernel(self.x[fit], self.y[fit], self.z[fit], config)
                mean = self.x[fit].mean(axis=0) if config["transform"] == "support_center" else np.zeros((2, 5))
                self.assertEqual(fold["fit_mean_sha256"], array_digest(mean))
                np.testing.assert_allclose(residuals[config_key(config)][held],
                                           predict_kernel(state, self.x[held]), atol=2e-14)
        # A changed held image cannot alter the fitting mean or the other image
        # in that same held group. This catches global pre-centering.
        changed = self.x.copy()
        changed[0] += 100.
        second, other_audits = leave_group_out_grid(changed, self.y, self.z, self.groups, configs)
        for config in configs:
            np.testing.assert_array_equal(
                residuals[config_key(config)][1], second[config_key(config)][1])
        self.assertEqual(audits["support_center"]["folds"][0]["fit_mean_sha256"],
                         other_audits["support_center"]["folds"][0]["fit_mean_sha256"])

    def test_missing_fold_label_state_returns_baseline_and_is_logged(self):
        labels = self.y.copy()
        labels[:, 0] = [1, 1, 0, 0, 0, 0, 0, 0]
        configs = [self.config, {**self.config, "kernel": "rbf", "tau": 20.}]
        predictions, audits = leave_group_out_grid(self.x, labels, self.z, self.groups, configs)
        for config in configs:
            np.testing.assert_array_equal(predictions[config_key(config)][:2, 0], 0.)
        self.assertEqual(audits["normalize"]["missing_state_rows_per_class"][0], 2)
        first = audits["normalize"]["folds"][0]
        self.assertEqual(first["zero_residual_classes"], [0])
        self.assertEqual(first["fit_positive_counts"], labels[2:].sum(axis=0).tolist())

    def test_centered_constant_vectors_and_independent_class_axes(self):
        constant = np.ones_like(self.x)
        for kernel in ("linear", "rbf"):
            config = {**self.config, "transform": "support_center", "kernel": kernel}
            state = fit_kernel(constant, self.y, self.z, config)
            self.assertTrue(np.isfinite(predict_kernel(state, constant)).all())
            np.testing.assert_array_equal(transform_features(constant, state["mean"]), 0.)
        config = {**self.config, "kernel": "rbf", "tau": 5.}
        state = fit_kernel(self.x, self.y, self.z, config)
        modified = self.x.copy()
        modified[:, 1] *= -1
        np.testing.assert_array_equal(predict_kernel(state, self.x)[:, 0],
                                      predict_kernel(state, modified)[:, 0])
        with self.assertRaises(TypeError):
            predict_kernel(state, self.x, labels=self.y)

    def test_export_reload_preserves_predictions_without_saving_labels(self):
        for kernel in ("linear", "rbf"):
            state = fit_kernel(self.x, self.y, self.z, {
                **self.config, "transform": "support_center", "kernel": kernel})
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "fit.npz"
                save_fit(path, state, "support_sha")
                with np.load(path, allow_pickle=False) as data:
                    self.assertNotIn("labels", data.files)
                    self.assertNotIn("support_labels", data.files)
                reloaded = load_fit(path)
                np.testing.assert_array_equal(predict_kernel(state, self.x),
                                              predict_kernel(reloaded, self.x))

    def test_selection_exact_tie_and_default_grid_family_count(self):
        args = parse_args(["--cache", "/cache", "--output", "/output"])
        grid = build_grid(args)
        self.assertEqual(len(grid), 24)
        self.assertEqual(set(map(family_name, grid)),
                         {"linear_raw", "linear_centered", "rbf_raw", "rbf_centered"})
        self.assertEqual(args.strengths, [0., .3, 1., 3., 10.])
        records = []
        for name, score, bce, strength in (
                ("zero", .9, .5, 0.), ("strong", .9, .3, 3.),
                ("tiny", .9, .3, .3), ("lower", np.nextafter(.9, 0), .1, 1.)):
            records.append({"candidate_id": name, "selection_score": score,
                            "support_oof_bce": {"weighted_bce": bce},
                            "config": {**self.config, "strength": strength}})
        self.assertEqual(choose_candidates(records, 4)[0]["candidate_id"], "tiny")
        records[-1]["selection_score"] = np.nextafter(.9, 1)
        self.assertEqual(choose_candidates(records, 4)[0]["candidate_id"], "lower")

    def test_deadline_and_invalid_inputs(self):
        with self.assertRaises(DeadlineReached):
            leave_group_out_grid(self.x, self.y, self.z, self.groups, [self.config], deadline=1.)
        with self.assertRaises(ValueError):
            leave_group_out_grid(self.x, self.y, self.z, ["a"] * 8, [self.config])
        with self.assertRaises(ValueError):
            fit_kernel(self.x, self.y * 2, self.z, self.config)
        with self.assertRaises(ValueError):
            kernel_matrix(self.x[:, 0], self.x[:, 0], "rbf", -1.)
        with self.assertRaises(SystemExit):
            parse_args(["--cache", "/cache", "--output", "/output", "--threads", "5"])

    def make_cache(self, root):
        cache, splits = root / "cache", root / "splits"
        cache.mkdir()
        (splits / "Toy").mkdir(parents=True)
        specs, identities, arrays = {}, {}, {}
        for split in ("support", "test"):
            prefix = "s" if split == "support" else "t"
            (splits / "Toy" / f"{split}.csv").write_text(split)
            metadata = {
                "name": "Toy", "data_root": str(root),
                "image_paths": [f"{prefix}{i}.png" for i in range(8)],
                "ids": [f"{prefix}{i}" for i in range(8)],
                "groups": self.groups if split == "support" else [f"t{i}" for i in range(8)],
                "class_names": ["a", "b"], "prompts": ["a", "b"],
            }
            specs[split] = {**metadata, "labels": self.y.copy()}
            identities[split] = fingerprint(metadata)
            (cache / f"{split}_metadata.json").write_text(json.dumps(metadata))
            arrays[split] = {"files": {}}
            values = {"class_local": self.x.astype(np.float32),
                      "base_logits": self.z.astype(np.float32)}
            for key, value in values.items():
                path = cache / f"{split}_{key}.npy"
                np.save(path, value)
                arrays[split]["files"][key] = {
                    "file": path.name, "sha256": sha256(path),
                    "shape": list(value.shape), "dtype": str(value.dtype)}
        np.save(cache / "support_labels.npy", self.y)
        manifest = {
            "status": "complete", "dataset": "Toy", "split_dir": str(splits),
            "class_names": ["a", "b"], "splits": arrays,
            "support_identity": identities["support"], "test_identity": identities["test"],
            "split_sha256": {str(splits / "Toy" / f"{split}.csv"):
                             sha256(splits / "Toy" / f"{split}.csv")
                             for split in ("support", "test")},
            "auxiliary_files": {path.name: {"sha256": sha256(path)} for path in (
                cache / "support_labels.npy", cache / "support_metadata.json",
                cache / "test_metadata.json")},
        }
        (cache / "manifest.json").write_text(json.dumps(manifest))
        return cache, specs

    def test_end_to_end_labels_only_after_selection_and_flipping_test_changes_no_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, specs = self.make_cache(root)
            selections, predictions = [], []
            for flip in (False, True):
                output = root / ("flipped" if flip else "original")
                args = parse_args([
                    "--cache", str(cache), "--output", str(output), "--dataset", "Toy",
                    "--linear-l2s", ".7", "--rbf-l2s", ".7", "--taus", "5",
                    "--strengths", "0", ".3", "1", "--threads", "1"])
                reads = []

                def load(directory, name, split):
                    if split == "test":
                        self.assertTrue((output / "selection.json").exists())
                        self.assertEqual(len(json.loads(
                            (output / "selection.json").read_text())["candidates"]), 4)
                        reads.append(split)
                        return {**specs[split], "labels": 1 - self.y if flip else self.y}
                    return specs[split]

                with patch("i2cl_data.load_spec", side_effect=load):
                    run(args)
                self.assertEqual(len(reads), 4)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(len(summary["results"]), 4)
                self.assertEqual(sum(item["support_selected"] for item in summary["results"]), 1)
                self.assertTrue(summary["results"][0]["support_selected"])
                self.assertFalse(summary["test_labels_used_in_fitting_or_selection"])
                selections.append([item["config"] for item in summary["results"]])
                predictions.append([np.load(output / item["candidate_id"] / "predictions.npy")
                                    for item in summary["results"]])
                self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
                for item in summary["results"]:
                    destination = output / item["candidate_id"]
                    self.assertTrue((destination / "oof_logits.npy").is_file())
                    self.assertTrue((destination / "predictions.csv").is_file())
                with patch("i2cl_data.load_spec", side_effect=load):
                    with self.assertRaises(FileExistsError):
                        run(args)
            self.assertEqual(selections[0], selections[1])
            for first, second in zip(*predictions):
                np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
