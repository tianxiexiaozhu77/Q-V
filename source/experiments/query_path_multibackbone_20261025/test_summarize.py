#!/usr/bin/env python3
"""Independent checks of clustering, pairing, macro means and control labels."""
import tempfile
from pathlib import Path
import json
import unittest

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

from summarize import (bootstrap, ci_record, scoped_metrics, summarize_model,
                       verify_prediction, validate_run_plan, MODEL_CONTROLS, TASKS, SEEDS, sha)


class StatisticsTest(unittest.TestCase):
    def test_cluster_weights_equal_expanded_patient_draws(self):
        patient = np.asarray(["a", "a", "b", "b", "b", "c", "c", "d"])
        y = np.asarray([0, 1, 0, 0, 1, 1, 1, 0])[:, None]
        scores = {"original": np.asarray([.2, .2, .4, .1, .4, .8, .6, .7])[:, None],
                  "readout_only": np.asarray([.1, .3, .5, .2, .3, .7, .9, .6])[:, None],
                  "query": np.asarray([.2, .4, .1, .2, .5, .7, .8, .3])[:, None]}
        plan = bootstrap.make_patient_draw_plan(patient, n_resamples=31, seed=91)
        result = bootstrap.paired_cluster_bootstrap(y, scores, patient, task_names=["task"], plan=plan)
        for draw, counts in enumerate(plan.counts):
            indices = np.concatenate([np.tile(np.flatnonzero(patient == pid), int(count))
                                      for pid, count in zip(plan.patient_ids, counts)])
            for branch, score in enumerate(scores.values()):
                labels = y[indices, 0]
                if len(np.unique(labels)) < 2:
                    self.assertTrue(np.isnan(result.replicate_metrics[draw, branch]).all())
                    continue
                precision, recall, _ = precision_recall_curve(labels, score[indices, 0])
                expected = [roc_auc_score(labels, score[indices, 0]), auc(recall, precision)]
                np.testing.assert_allclose(result.replicate_metrics[draw, branch, 0, :2], expected, atol=1e-14)

    def test_macro_and_five_seed_ci_preserve_pairing(self):
        # Seeds differ by a common shift in both arms. A paired difference has
        # zero SD despite nonzero per-arm support SD.
        points = np.zeros((5, 3, 3, 3), dtype=float)
        for seed in range(5):
            points[seed, 0] = .5
            points[seed, 1] = .55 + .02 * seed
            points[seed, 2] = points[seed, 1] + np.asarray([.03, -.06, .09])[:, None]
        draws = np.repeat(points[:, None], 11, axis=1)
        # Shared replicate noise must cancel in contrasts.
        draws += np.linspace(-.01, .01, 11)[None, :, None, None, None]
        means, pairs, per_seed, seed_pairs = summarize_model("CARZero", MODEL_CONTROLS["CARZero"], points, draws)
        self.assertEqual(len(means), 3 * 4 * 3)
        self.assertEqual(len(per_seed), 5 * 3 * 4 * 3)
        macro = next(r for r in pairs if r["task"] == "macro_three_tasks" and
                     r["comparison"] == "query_minus_fixed_aggregation" and r["metric"] == "auroc")
        self.assertAlmostEqual(macro["mean_delta_pp"], 2)
        self.assertAlmostEqual(macro["support_seed_delta_sd_pp"], 0)
        self.assertAlmostEqual(macro["ci_low_pp"], 2)
        self.assertAlmostEqual(macro["ci_high_pp"], 2)
        self.assertTrue(all(r["control_name"] == "fixed_aggregation" for r in pairs))
        self.assertFalse(any("readout" in r["comparison"] for r in pairs))
        negative = next(r for r in pairs if r["task"] == "dlv" and
                        r["comparison"] == "query_minus_fixed_aggregation" and r["metric"] == "auroc")
        self.assertEqual(negative["negative_gain_seed_count"], 5)
        self.assertAlmostEqual(negative["mean_delta_pp"], -6)

    def test_invalid_task_is_not_dropped_from_macro(self):
        points = np.zeros((5, 3, 3, 3))
        draws = np.zeros((5, 7, 3, 3, 3))
        draws[:, 2, :, 1, :] = np.nan
        _, scoped = scoped_metrics(points, draws)
        self.assertTrue(np.isnan(scoped[:, 2, :, -1, :]).all())
        ci = ci_record(0, scoped.mean(axis=0)[:, 0, -1, 0])
        self.assertEqual(ci["valid_replicates"], 6)
        self.assertEqual(ci["invalid_replicates"], 1)

    def test_reject_misaligned_patients_accept_native_logits(self):
        expected = dict(image_paths=np.asarray(["a", "b"]), patient_ids=np.asarray(["p", "q"]),
                        labels=np.asarray([[1, 0, 1], [0, 1, 1]]))
        data = dict(expected)
        for branch in ("original", "readout_only", "query"):
            data[branch] = np.asarray([[7, -4, 1], [-2, 3, 4]], dtype=float)
        verify_prediction(data, expected, ("original", "readout_only", "query"))
        data["patient_ids"] = np.asarray(["q", "p"])
        with self.assertRaisesRegex(ValueError, "patient_ids"):
            verify_prediction(data, expected, ("original", "readout_only", "query"))

    def test_actual_worker_metadata_keys_and_seed_are_required(self):
        expected = dict(image_paths=np.asarray(["a", "b"]), patient_ids=np.asarray(["p", "q"]),
                        labels=np.asarray([[1, 0, 1], [0, 1, 1]]), task_keys=np.asarray(TASKS),
                        class_names=np.asarray(["CSLV name", "CDLV name", "CCOM name"]))
        data = dict(expected, seed=np.asarray(42))
        for branch in ("original", "readout_only", "query"):
            data[branch] = np.zeros((2, 3))
        verify_prediction(data, expected, ("original", "readout_only", "query"), expected_seed=42)
        for field in ("task_keys", "class_names"):
            changed = dict(data)
            changed[field] = changed[field][::-1]
            with self.assertRaisesRegex(ValueError, field):
                verify_prediction(changed, expected, ("original", "readout_only", "query"), expected_seed=42)
        with self.assertRaisesRegex(ValueError, "support seed"):
            verify_prediction(data, expected, ("original", "readout_only", "query"), expected_seed=43)

    def test_run_plan_preserves_all_models_and_distinct_controls(self):
        with tempfile.TemporaryDirectory(prefix="query_path_plan_test_") as temporary:
            folder = Path(temporary)
            protocol = folder / "protocol.json"
            protocol.write_text("{}")
            path = folder / "plan.json"
            models = ["AlphaRAD", "CARZero"]
            plan = dict(models=models, protocol=str(protocol), protocol_sha256=sha(protocol),
                        support_seeds=list(SEEDS), tasks=list(TASKS), eta=.3, views=["identity"],
                        all_models_tasks_seeds_retained=True, statistics_after_all_jobs=True,
                        primary_contrast={"AlphaRAD": "Q - R (readout only)",
                                          "CARZero": "Q - F (fixed visual attention outputs)"})
            path.write_text(json.dumps(plan))
            experiment = dict(model_controls=MODEL_CONTROLS)
            validate_run_plan(path, models, protocol, experiment)
            with self.assertRaisesRegex(ValueError, "model scope"):
                validate_run_plan(path, ["AlphaRAD"], protocol, experiment)
            plan["primary_contrast"]["CARZero"] = "Q - R (readout only)"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "controls"):
                validate_run_plan(path, models, protocol, experiment)


if __name__ == "__main__":
    unittest.main()
