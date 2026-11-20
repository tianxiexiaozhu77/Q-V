"""CPU checks against dense equations and support-only selection invariants."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from neighbor_methods import (
    LINEAR_PROBE_MAX_ITER, LINEAR_PROBE_TOL, fit_project_mix_binary,
    run_neighbor_methods,
)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class NeighborMethodsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261025)
        self.x = unit(self.rng.normal(size=(10, 17)))
        self.y = np.asarray([0] * 5 + [1] * 5)
        self.pos, self.neg = unit(self.rng.normal(size=(2, 17)))
        self.test = unit(self.rng.normal(size=(12, 17)))
        self.groups = [f"patient-{i}" for i in range(10)]
        self.folds = [
            {"validation": [i, i + 5], "train": [j for j in range(10) if j not in (i, i + 5)]}
            for i in range(5)
        ]

    def arguments(self):
        return dict(support_features=self.x, labels=self.y[:, None], groups=self.groups,
                    folds=self.folds, queries_positive=self.pos[None],
                    queries_negative=self.neg[None], test_features=self.test)

    def test_complete_cmp_matches_independent_dense_equations(self):
        model = fit_project_mix_binary(self.x, self.y, self.pos, self.neg)
        means = np.stack([self.x[self.y == label].mean(0) for label in (0, 1)])
        texts = np.stack([self.neg, self.pos])
        u, singular, _ = np.linalg.svd(texts.T, full_matrices=False)
        k = np.searchsorted(np.cumsum(singular ** 2) / np.sum(singular ** 2), .99) + 1
        projector = u[:, :k] @ u[:, :k].T
        residual = self.x - means[self.y]
        covariance = np.cov(residual.T, ddof=1)
        matrix = (len(self.x) - 1) * covariance + np.trace(covariance) * np.eye(self.x.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        lam, gamma = .37, .61
        mixed = lam * (means @ projector) + (1 - lam) * texts
        semantic = unit(self.test @ projector) @ unit(mixed).T
        image = unit(self.test @ whitening) @ unit(means @ whitening).T
        expected = semantic + gamma * image
        np.testing.assert_allclose(model.predict_margin(self.test, lam, gamma),
                                   expected[:, 1] - expected[:, 0], atol=2e-14, rtol=1e-12)

    def test_rank_one_follows_energy_rule_and_keeps_full_text_norm(self):
        neg = np.zeros(17)
        pos = np.zeros(17)
        neg[0], neg[1] = 1, -.02
        pos[0], pos[1] = 1, .02
        neg, pos = unit(neg[None])[0], unit(pos[None])[0]
        model = fit_project_mix_binary(self.x, self.y, pos, neg)
        self.assertEqual(model.audit["text_rank"], 1)
        projector = model.basis @ model.basis.T
        lam = .6
        mixed = lam * model.means @ projector + (1 - lam) * np.stack([neg, pos])
        expected = unit(self.test @ projector) @ unit(mixed).T
        np.testing.assert_allclose(model.semantic_scores(self.test, lam), expected, atol=2e-14)
        # All images with positive projected sign must share the same semantic
        # margin in a rank-one subspace; do not let floating noise rank them.
        selected = self.test[(self.test @ model.basis)[:, 0] > 0]
        scores = model.semantic_scores(selected, lam)
        np.testing.assert_array_equal(scores, np.repeat(scores[:1], len(scores), axis=0))

    def test_binary_label_swap_reverses_cmp_margin(self):
        forward = fit_project_mix_binary(self.x, self.y, self.pos, self.neg)
        reverse = fit_project_mix_binary(self.x, 1 - self.y, self.neg, self.pos)
        np.testing.assert_allclose(forward.predict_margin(self.test, .4, .7),
                                   -reverse.predict_margin(self.test, .4, .7), atol=2e-14)

    def test_rotation_equivariance(self):
        rotation, _ = np.linalg.qr(self.rng.normal(size=(17, 17)))
        base = fit_project_mix_binary(self.x, self.y, self.pos, self.neg)
        rotated = fit_project_mix_binary(self.x @ rotation, self.y,
                                         self.pos @ rotation, self.neg @ rotation)
        np.testing.assert_allclose(base.predict_margin(self.test, .6, .3),
                                   rotated.predict_margin(self.test @ rotation, .6, .3), atol=2e-14)

    def test_test_inputs_never_change_fit_or_hyperparameters(self):
        args = self.arguments()
        first = run_neighbor_methods(**args)
        args["test_features"] = unit(self.rng.normal(size=(5, 17)))
        other = run_neighbor_methods(**args)
        for name in ("project_mix", "linear_probe"):
            a = copy.deepcopy(first["fit_info"]["methods"][name])
            b = copy.deepcopy(other["fit_info"]["methods"][name])
            for info in (a, b):
                info.pop("fit_seconds")
                info.pop("prediction_seconds")
            self.assertEqual(a, b)
        json.dumps(first["fit_info"], allow_nan=False)

    def test_linear_probe_predictions_equal_selected_sklearn_fit(self):
        from sklearn.linear_model import LogisticRegression
        out = run_neighbor_methods(**self.arguments(), methods=["linear_probe"])
        selected = out["fit_info"]["methods"]["linear_probe"]["selected"]
        reference = LogisticRegression(C=selected["C"], penalty="l2", solver="lbfgs",
                                       fit_intercept=True, class_weight=None,
                                       tol=LINEAR_PROBE_TOL, max_iter=LINEAR_PROBE_MAX_ITER,
                                       random_state=0).fit(self.x, self.y)
        np.testing.assert_array_equal(out["scores"]["linear_probe"][:, 0],
                                      reference.decision_function(self.test))

    def test_folds_reject_patient_leakage_before_fit(self):
        args = self.arguments()
        args["groups"] = ["same_patient"] * 10
        with self.assertRaisesRegex(ValueError, "patient/group crosses"):
            run_neighbor_methods(**args)
        args = self.arguments()
        args["folds"] = self.folds[:4]
        with self.assertRaisesRegex(ValueError, "Exactly five"):
            run_neighbor_methods(**args)

    def test_degenerate_covariance_fails_explicitly(self):
        features = np.tile(np.eye(1, 17)[0], (10, 1))
        with self.assertRaisesRegex(ValueError, "Zero support covariance"):
            fit_project_mix_binary(features, self.y, self.pos, self.neg)


if __name__ == "__main__":
    unittest.main()
