"""CPU equivalence, group isolation, zero identity, selection/label-separation."""
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import overnight_pairwise_residual as worker
from batch_metrics import compute_metrics


class PairwiseTests(unittest.TestCase):
    def test_graph_equals_explicit_pairs(self):
        labels = np.array([1, 0, 0, 1, 0])
        rows = []
        for p in np.flatnonzero(labels):
            for n in np.flatnonzero(labels == 0):
                row = np.zeros(len(labels))
                row[p], row[n] = 1., -1.
                rows.append(row)
        d = np.asarray(rows)
        laplacian, q = worker.pair_system(labels)
        np.testing.assert_allclose(laplacian, d.T @ d / len(d), atol=1e-14)
        np.testing.assert_allclose(q, d.mean(axis=0), atol=1e-14)

    def test_closed_form_equals_explicit_primal_pair_ridge(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(9, 5))
        b = rng.normal(size=9)
        y = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0])
        for penalty in worker.PENALTIES:
            solutions = worker.class_solutions(x, y, b, [penalty])
            for margin in worker.MARGINS:
                rows, targets = [], []
                for p in np.flatnonzero(y):
                    for n in np.flatnonzero(y == 0):
                        rows.append(x[p] - x[n])
                        targets.append(margin - (b[p] - b[n]))
                d, target = np.asarray(rows), np.asarray(targets)
                expected = np.linalg.solve(d.T @ d / len(d) + penalty * np.eye(5),
                                           d.T @ target / len(d))
                actual = margin * solutions[penalty][:, 0] + solutions[penalty][:, 1]
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_rank_deficient_features_and_constant_logit_shift(self):
        x = np.array([[1., 2., 3.], [1., 2., 3.], [-1., -2., -3.], [0., 0., 0.]])
        y, b = np.array([1, 0, 0, 1]), np.array([.3, -.2, .4, .1])
        a = worker.class_solutions(x, y, b, [.001])[.001]
        c = worker.class_solutions(x, y, b + 100, [.001])[.001]
        np.testing.assert_allclose(a, c, atol=1e-9, rtol=1e-9)
        self.assertTrue(np.isfinite(a).all())

    def test_grouped_oof_equals_independent_fold_fits_missing_class_zero(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(6, 2, 4))
        labels = np.array([[1, 1], [1, 1], [0, 0], [0, 1], [0, 0], [0, 0]])
        b = rng.normal(size=labels.shape)
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        scores, audit = worker.grouped_oof_grid(x, labels, b, groups, [.25, 1.], [.01, .1])
        for fold in audit["folds"]:
            held, fit = fold["held_indices"], fold["fit_indices"]
            self.assertFalse(set(groups[held]) & set(groups[fit]))
            for (margin, penalty), predictions in scores.items():
                state = worker.fit_pairwise(x[fit], labels[fit], b[fit], margin, penalty)
                np.testing.assert_allclose(predictions[held],
                                           worker.predict_pairwise(state, x[held]), atol=1e-10)
        np.testing.assert_array_equal(scores[.25, .01][:2, 0], 0)
        self.assertEqual(audit["missing_state_rows_per_class"], [2, 0])

    def test_holdout_labels_do_not_enter_fitted_state(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(8, 1, 5))
        labels = np.array([[1], [1], [0], [0], [1], [0], [1], [0]])
        changed = labels.copy()
        changed[:2] = 0
        b, groups = rng.normal(size=(8, 1)), np.repeat(["a", "b", "c", "d"], 2)
        before, _ = worker.grouped_oof_grid(x, labels, b, groups, [1.], [.1])
        after, _ = worker.grouped_oof_grid(x, changed, b, groups, [1.], [.1])
        np.testing.assert_array_equal(before[1., .1][:2], after[1., .1][:2])

    def test_pair_risk_never_pairs_different_classes_and_fixed_margin(self):
        y = np.array([[1, 0], [0, 1]])
        scores = np.array([[1., 10.], [0., 12.]])
        risk = worker.pairwise_oof_risk(y, scores)
        self.assertEqual(risk["macro_squared_hinge_risk"], 0.)
        self.assertEqual(risk["pairs_per_class"], [1, 1])
        self.assertEqual(risk["reference_margin"], 1.)
        zero = worker.pairwise_oof_risk(y, np.zeros_like(scores))
        self.assertEqual(zero["macro_squared_hinge_risk"], 1.)

    def test_selection_exact_ties_use_risk_then_small_strength(self):
        def record(score, risk, strength):
            return {"selection_score": score,
                    "support_oof_pair_risk": {"macro_squared_hinge_risk": risk},
                    "config": {"parameter": .1, "margin": 1., "strength": strength}}
        primary = record(.9, 100., 4.)
        tied = record(.8, .1, 1.)
        weaker = record(.8, .1, .5)
        bad_risk = record(.8, .2, 0.)
        self.assertEqual(sorted([bad_risk, tied, primary, weaker], key=worker.candidate_order),
                         [primary, weaker, tied, bad_risk])

    def test_stop_and_deadline_are_checked(self):
        with self.assertRaises(worker.StopRequested):
            worker.check_limits(0.)
        with tempfile.TemporaryDirectory() as root:
            stop = Path(root) / "STOP"
            stop.touch()
            with self.assertRaises(worker.StopRequested):
                worker.check_limits(stop_file=stop)

    def test_pipeline_test_label_flip_cannot_change_fit_selection_predictions(self):
        from argparse import Namespace
        rng = np.random.default_rng(1)
        x = rng.normal(size=(6, 2, 4))
        labels = np.array([[1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [0, 1]])
        b, tx = rng.normal(size=(6, 2)), rng.normal(size=(4, 2, 4))
        tb = rng.normal(size=(4, 2))
        test_labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}")
            class FakeCache:
                dataset, path, split_dir = "Fixture", manifest_path, root
                metadata = {"support": {"csv_sha256": "support"},
                            "test": {"csv_sha256": "test"}}
                manifest = {"checkpoint": {}, "support_identity": "fixed"}
                def support(self):
                    return {"labels": labels, "groups": [str(i) for i in range(6)],
                            "class_names": ["A", "B"]}
                def array(self, split, name):
                    return {("support", "class_local"): x, ("support", "baseline_logits"): b,
                            ("test", "class_local"): tx, ("test", "baseline_logits"): tb}[split, name]
            artifacts = []
            for index, truth in enumerate((test_labels, 1 - test_labels)):
                output = root / str(index)
                def evaluate(split_dir, dataset, predictions, base, identity, destination):
                    self.assertTrue((output / "selection.json").exists())
                    self.assertTrue((destination / "pairwise_residual.npz").exists())
                    self.assertTrue((destination / "predictions.npy").exists())
                    spec = {"labels": truth, "class_names": ["A", "B"], "name": "Fixture"}
                    base_metrics = compute_metrics(spec, base)
                    metrics = compute_metrics(spec, predictions)
                    return {"baseline_metrics": base_metrics, "metrics": metrics,
                            "delta": worker.metric_delta(base_metrics, metrics)}
                args = Namespace(cache=str(manifest_path), dataset="Fixture", output=str(output),
                                 deadline=None, stop_file=root / "STOP", chunk_size=2)
                with patch.object(worker, "FrozenCache", return_value=FakeCache()), \
                     patch.object(worker, "evaluate_test_only", side_effect=evaluate):
                    result = worker.run(args)
                selected = json.loads((output / "selection.json").read_text())["selected"]
                destination = output / result["candidate_id"]
                artifacts.append((selected, np.load(destination / "pairwise_residual.npz")["weight"],
                                  np.load(destination / "predictions.npy")))
            self.assertEqual(artifacts[0][0], artifacts[1][0])
            np.testing.assert_array_equal(artifacts[0][1], artifacts[1][1])
            np.testing.assert_array_equal(artifacts[0][2], artifacts[1][2])


if __name__ == "__main__":
    unittest.main()
