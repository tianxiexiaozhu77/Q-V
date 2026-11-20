"""Independent sklearn/explicit repeated-patient checks for bootstrap metrics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from cluster_bootstrap import (PatientDrawPlan, binary_weighted_metrics,
                               load_patient_draw_plan, make_patient_draw_plan,
                               paired_cluster_bootstrap, save_bootstrap,
                               save_patient_draw_plan)


def sklearn_metrics(y, p):
    if len(np.unique(y)) != 2:
        return np.full(3, np.nan)
    precision, recall, _ = precision_recall_curve(y, p)
    return np.asarray([roc_auc_score(y, p), auc(recall, precision),
                       average_precision_score(y, p)])


class PatientClusterBootstrapTests(unittest.TestCase):
    def setUp(self):
        # Unequal cluster sizes; p1 has mixed labels, so this is not a patient-
        # averaged classification problem. Tied scores exercise PR endpoints.
        self.patients = np.asarray(["p1", "p1", "p1", "p2", "p3", "p3", "p4"])
        self.labels = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [0, 0], [1, 1]])
        self.scores = {
            "original": np.asarray([[.2, .9], [.8, .9], [.8, .4], [.2, .4], [.5, .2], [.2, .4], [.7, .2]]),
            "query": np.asarray([[.4, .8], [.5, .6], [.9, .5], [.1, .2], [.3, .4], [.3, .2], [.8, .8]])}

    def test_weighted_metrics_equal_expanded_sklearn_including_ties_and_zero_weights(self):
        rng = np.random.default_rng(20)
        for _ in range(40):
            y = rng.integers(0, 2, size=23)
            score = rng.integers(0, 5, size=23).astype(float)
            w = rng.integers(0, 4, size=23)
            ix = np.repeat(np.arange(len(y)), w)
            np.testing.assert_allclose(binary_weighted_metrics(y, score, w),
                                       sklearn_metrics(y[ix], score[ix]), atol=2e-15, rtol=0)

    def test_every_replicate_equals_explicit_whole_patient_sampling(self):
        plan = make_patient_draw_plan(self.patients, n_resamples=37, seed=3)
        result = paired_cluster_bootstrap(self.labels, self.scores, self.patients,
                                         task_names=["A", "B"], plan=plan)
        for r, counts in enumerate(plan.counts):
            expanded = np.concatenate([np.tile(np.flatnonzero(self.patients == pid), int(count))
                                       for pid, count in zip(plan.patient_ids, counts)])
            for ib, scores in enumerate(self.scores.values()):
                for c in range(2):
                    expected = sklearn_metrics(self.labels[expanded, c], scores[expanded, c])
                    np.testing.assert_allclose(result.replicate_metrics[r, ib, c], expected,
                                               atol=2e-15, rtol=0, equal_nan=True)
        paired = result.replicate_metrics[:, 1, 0, 0] - result.replicate_metrics[:, 0, 0, 0]
        valid = paired[np.isfinite(paired)]
        got = result.summary["paired_differences"]["query"]["per_task"]["A"]["auroc"]
        np.testing.assert_allclose(got["ci"], np.quantile(valid, [.025, .975]), atol=1e-15)
        self.assertEqual(result.summary["test_counts"]["A"]["patients_with_both_labels"], 1)

    def test_patient_order_invariance_and_shared_draws(self):
        plan = make_patient_draw_plan(self.patients, n_resamples=31, seed=19)
        permutation = np.asarray([6, 3, 0, 5, 2, 4, 1])
        reordered_plan = make_patient_draw_plan(self.patients[permutation], n_resamples=31, seed=19)
        self.assertEqual(plan.sha256, reordered_plan.sha256)
        first = paired_cluster_bootstrap(self.labels, self.scores, self.patients, plan=plan)
        second = paired_cluster_bootstrap(self.labels[permutation],
                    {k: v[permutation] for k, v in self.scores.items()}, self.patients[permutation], plan=plan)
        np.testing.assert_allclose(first.replicate_metrics, second.replicate_metrics,
                                   atol=2e-15, rtol=0, equal_nan=True)

    def test_invalid_draws_reported_and_not_silently_scored(self):
        plan = PatientDrawPlan(np.asarray(["negative", "positive"]),
                               np.asarray([[2, 0], [1, 1], [0, 2]], dtype=np.uint32), 1)
        scores = {"original": np.asarray([[.1], [.9]]), "same": np.asarray([[.1], [.9]])}
        result = paired_cluster_bootstrap(np.asarray([[0], [1]]), scores,
                                          ["negative", "positive"], task_names=["task"], plan=plan)
        summary = result.summary["paired_differences"]["same"]["per_task"]["task"]["auprc"]
        self.assertEqual(summary["invalid_replicates"], 2)
        self.assertEqual(summary["valid_fraction"], 1 / 3)
        self.assertEqual(summary["ci"], [0.0, 0.0])
        self.assertTrue(np.isnan(result.replicate_metrics[0]).all())
        json.dumps(result.summary, allow_nan=False)

    def test_auprc_is_not_average_precision(self):
        values = binary_weighted_metrics(np.asarray([0, 1, 0, 1]), np.asarray([.9, .8, .7, .6]))
        self.assertNotAlmostEqual(values[1], values[2])
        np.testing.assert_allclose(values, sklearn_metrics([0, 1, 0, 1], [.9, .8, .7, .6]))

    def test_plan_and_result_roundtrip_without_pickle(self):
        plan = make_patient_draw_plan(self.patients, n_resamples=10, seed=9)
        result = paired_cluster_bootstrap(self.labels, self.scores, self.patients, plan=plan)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draw_plan.npz"
            save_patient_draw_plan(plan, path)
            restored = load_patient_draw_plan(path)
            self.assertEqual(plan.sha256, restored.sha256)
            np.testing.assert_array_equal(plan.counts, restored.counts)
            save_bootstrap(result, Path(tmp) / "result")
            with np.load(Path(tmp) / "result/bootstrap_metrics.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["replicate_metrics"], result.replicate_metrics)
                self.assertEqual(saved["metric_names"].tolist(), ["auroc", "auprc", "average_precision"])
            self.assertEqual(json.loads((Path(tmp) / "result/bootstrap_summary.json").read_text()), result.summary)

    def test_wrong_patient_universe_and_invalid_weight_rejected(self):
        plan = make_patient_draw_plan(["q1", "q2"], n_resamples=3)
        with self.assertRaisesRegex(ValueError, "patient universe"):
            paired_cluster_bootstrap(self.labels, self.scores, self.patients, plan=plan)
        with self.assertRaisesRegex(ValueError, "Weights"):
            binary_weighted_metrics(np.asarray([0, 1]), np.asarray([.1, .9]), np.asarray([-1, 1]))
        with self.assertRaisesRegex(ValueError, "Missing"):
            make_patient_draw_plan(["p1", None])


if __name__ == "__main__":
    unittest.main()
