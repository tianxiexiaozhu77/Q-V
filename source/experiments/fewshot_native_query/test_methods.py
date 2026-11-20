"""CPU checks for upstream formulas, group isolation, and test-blind selection."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import methods


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def data(seed=3):
    rng = np.random.default_rng(seed)
    n, c, dim = 12, 2, 7
    x = unit(rng.normal(size=(n, dim)))
    y = np.asarray([[i % 2, (i // 2) % 2] for i in range(n)], dtype=float)
    return dict(support_features=x, support_local=unit(rng.normal(size=(n, c, dim))),
                labels=y, support_logits=rng.normal(size=(n, c)),
                groups=[f"patient{i // 2}" for i in range(n)],
                test_features=unit(rng.normal(size=(5, dim))),
                test_local=unit(rng.normal(size=(5, c, dim))),
                test_logits=rng.normal(size=(5, c)))


class FormulaTests(unittest.TestCase):
    def test_tip_matches_two_class_one_hot_source_formula(self):
        d = data()
        beta = 5.0
        actual = methods.tip_margin(d["test_features"], d["support_features"], d["labels"], beta)
        cache = np.exp(-beta + beta * (d["test_features"] @ d["support_features"].T))
        expected = []
        for c in range(d["labels"].shape[1]):
            one_hot = np.eye(2)[d["labels"][:, c].astype(int)]
            pair = cache @ one_hot
            expected.append(pair[:, 1] - pair[:, 0])
        assert_allclose(actual, np.stack(expected, 1), atol=1e-14)

    def test_gda_ks_matches_dense_author_formula_both_solver_paths(self):
        rng = np.random.default_rng(91)
        for n, dim in ((10, 37), (12, 4)):
            x = unit(rng.normal(size=(n, dim)))
            y = np.asarray([[i % 2, int(i % 3 == 0)] for i in range(n)])
            actual_w, actual_b, info = methods.fit_gda_ks(x, y)
            for c in range(y.shape[1]):
                means = np.stack([x[y[:, c] == k].mean(0) for k in (0, 1)])
                residual = np.concatenate([x[y[:, c] == k] - means[k] for k in (0, 1)])
                cov = np.cov(residual.T)
                inv = dim * np.linalg.pinv((n - 1) * cov + np.trace(cov) * np.eye(dim))
                w = inv @ means.T
                b = np.log(np.full(2, 0.5)) - np.einsum("nd,dc,nc->n", means, inv, means) / 2
                assert_allclose(actual_w[:, c], w[:, 1] - w[:, 0], atol=2e-11, rtol=2e-11)
                assert_allclose(actual_b[c], b[1] - b[0], atol=2e-11, rtol=2e-11)
            self.assertEqual(info[0]["solver"], "woodbury" if n < dim else "direct_small_dimension")

    def test_zero_covariance_has_author_pseudoinverse_not_hidden_ridge(self):
        x = np.tile([[1.0, 0.0, 0.0]], (4, 1))
        w, b, info = methods.fit_gda_ks(x, np.asarray([[0], [0], [1], [1]]))
        assert_array_equal(w, np.zeros((3, 1)))
        assert_array_equal(b, np.zeros(1))
        self.assertEqual(info[0]["solver"], "zero_covariance_pseudoinverse")

    def test_global_and_local_prototype_semantics_differ(self):
        x = np.asarray([[1.0, 0], [0.6, 0.8], [0, 1.0], [-0.6, 0.8]])
        y = np.asarray([[1], [1], [0], [0]])
        pm, nm = x[:2].mean(0), x[2:].mean(0)
        assert_allclose(methods.prototype_direction(x, y), (unit(pm) - unit(nm))[None])
        assert_allclose(methods.prototype_direction(x[:, None, :], y, local=True), unit(pm - nm)[None])


class IsolationTests(unittest.TestCase):
    def test_fold_rejects_patient_leakage_and_missing_coverage(self):
        with self.assertRaisesRegex(ValueError, "crosses"):
            methods.validate_folds(["a", "a", "b", "b"], [
                {"train": [0, 2], "validation": [1, 3]},
                {"train": [1, 3], "validation": [0, 2]},
            ])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            methods.validate_folds(["a", "b", "c"], [
                {"train": [1, 2], "validation": [0]},
                {"train": [1, 2], "validation": [0]},
            ])
        with self.assertRaisesRegex(ValueError, "integer"):
            methods.validate_folds(["a", "b"], [
                {"train": [1.1], "validation": [0]},
                {"train": [0], "validation": [1]},
            ])

    def test_test_features_and_scores_cannot_change_hyperparameters(self):
        d = data()
        before = methods.run_score_methods(**d)
        changed = dict(d)
        changed["test_features"] = -d["test_features"]
        changed["test_local"] = -d["test_local"]
        changed["test_logits"] = d["test_logits"] * 1e6
        after = methods.run_score_methods(**changed)
        self.assertEqual(before["fit_info"], after["fit_info"])
        for name in methods.DEFAULT_METHODS:
            self.assertEqual(before["scores"][name].shape, (5, 2))
            self.assertTrue(np.isfinite(before["scores"][name]).all())
        # Passing test labels is impossible through the public API.
        with self.assertRaises(TypeError):
            methods.run_score_methods(**d, test_labels=np.zeros((5, 2)))

    def test_every_gda_oof_fit_sees_only_training_rows(self):
        d = data()
        folds = methods.validate_folds(d["groups"])
        fitted = []
        original = methods.fit_gda_ks
        def spy(x, y):
            fitted.append((x.copy(), y.copy()))
            return original(x, y)
        with patch.object(methods, "fit_gda_ks", side_effect=spy):
            result = methods.run_score_methods(**d, folds=folds, methods=["gda_ovr_ks_oof"])
        for fold, (x, y) in zip(folds, fitted):
            assert_array_equal(x, d["support_features"][fold["train"]])
            assert_array_equal(y, d["labels"][fold["train"]])
        self.assertGreaterEqual(len(fitted), len(folds))
        self.assertLessEqual(len(fitted), len(folds) + 1)
        self.assertEqual(result["fit_info"]["gda_ovr_ks_oof"]["fold_type"], "provided_shared_group_partition")

    def test_invalid_binary_fold_causes_explicit_baseline_fallback(self):
        d = data()
        d["labels"] = d["labels"].copy()
        d["labels"][:, 0] = 0
        d["labels"][0, 0] = 1
        result = methods.run_score_methods(**d)
        for name in methods.DEFAULT_METHODS:
            if not name.endswith("_oof"):
                continue
            fit = result["fit_info"][name]
            self.assertFalse(fit["all_folds_fit_eligible"])
            self.assertEqual(fit["selection_status"], "no_selection_invalid_folds_baseline_fallback")
            self.assertEqual(fit["selected_parameters"]["alpha"], 0)
            assert_array_equal(result["scores"][name], d["test_logits"])
            for candidate in fit["candidates"]:
                if candidate["parameters"]["alpha"]:
                    self.assertIsNone(candidate["oof_macro_auprc"])

    def test_tied_perfect_oof_prefers_zero_and_exact_native_scores(self):
        d = data()
        d["support_logits"] = 2 * d["labels"] - 1
        result = methods.run_score_methods(**d)
        for name in methods.DEFAULT_METHODS:
            if name.endswith("_oof"):
                self.assertEqual(result["fit_info"][name]["selected_parameters"]["alpha"], 0)
                assert_array_equal(result["scores"][name], d["test_logits"])
        json.dumps(result["fit_info"], allow_nan=False)

    def test_prototype_oof_metrics_match_independently_refit_fold_predictions(self):
        d = data()
        folds = methods.validate_folds(d["groups"])
        residual = np.zeros_like(d["labels"])
        for fold in folds:
            tr, va = fold["train"], fold["validation"]
            x, y = d["support_features"][tr], d["labels"][tr]
            pos = y.T @ x / y.sum(0)[:, None]
            neg = (1 - y).T @ x / (1 - y).sum(0)[:, None]
            residual[va] = d["support_features"][va] @ (unit(pos) - unit(neg)).T
        result = methods.run_score_methods(**d, methods=["prototype_score_mix_oof"], folds=folds)
        for candidate in result["fit_info"]["prototype_score_mix_oof"]["candidates"]:
            score = d["support_logits"] + candidate["parameters"]["alpha"] * residual
            expected = methods.macro_trapezoid_auprc(d["labels"], score)
            self.assertAlmostEqual(candidate["oof_macro_auprc"], expected, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
