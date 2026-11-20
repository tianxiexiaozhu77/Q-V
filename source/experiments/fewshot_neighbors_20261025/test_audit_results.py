"""Synthetic checks for the post-run auditor; no real test labels are opened."""
import copy
import unittest

import numpy as np

import audit_results as audit


class ResultsAuditTests(unittest.TestCase):
    def test_every_summary_axis_and_interval(self):
        # The producer is used only to construct synthetic records for this
        # schema integration test. The audit executable never imports it.
        from summarize import summarize_model
        rng = np.random.default_rng(191)
        summary = {k: [] for k in ("mean_sd_results", "paired_comparisons",
                                  "per_seed_results", "per_seed_paired_comparisons")}
        points_by_model, draws_by_model = {}, {}
        for model in audit.MODELS:
            points = rng.uniform(.2, .9, size=(5, 18, 3, 3))
            draws = points[:, None] + rng.normal(0, .01, size=(5, 23, 18, 3, 3))
            for key, rows in zip(summary, summarize_model(model, points, draws)):
                summary[key].extend(rows)
            points_by_model[model] = np.concatenate((points, points.mean(2, keepdims=True)), axis=2)
            draws_by_model[model] = np.concatenate((draws, draws.mean(3, keepdims=True)), axis=3)
        report = audit.audit_rows(summary, points_by_model, draws_by_model)
        self.assertEqual(report["rows_checked"], 6192)
        self.assertLess(report["maximum_absolute_error_percent_or_pp"], 1e-12)
        broken = copy.deepcopy(summary)
        broken["paired_comparisons"][-1]["ci_low_pp"] += .1
        with self.assertRaisesRegex(ValueError, "interval differs"):
            audit.audit_rows(broken, points_by_model, draws_by_model)

    def test_weighted_patient_replicates_match_expanded_images_with_ties(self):
        labels = np.asarray([0, 1, 1, 0, 1, 0])
        scores = np.asarray([.1, .4, .4, .2, .7, .7])
        counts = np.asarray([2, 2, 0, 0, 3, 3])
        indices = np.repeat(np.arange(6), counts)
        np.testing.assert_allclose(audit.sklearn_metrics(labels, scores, counts),
                                   audit.sklearn_metrics(labels[indices], scores[indices]), atol=1e-15)

    def test_real_wrapper_metadata_is_accepted_and_misselection_rejected(self):
        from test_evaluate import FairViewComparisonTests
        FairViewComparisonTests.setUpClass()
        kw = FairViewComparisonTests.kw
        rows = [{"image_path": f"support-{i}", "patient_id": kw["groups"][i],
                 "label": int(kw["labels"][i, 0])} for i in range(10)]
        fitted = {"support_images": [r["image_path"] for r in rows],
                  "support_patient_ids": [r["patient_id"] for r in rows],
                  "support_labels": [r["label"] for r in rows],
                  "folds": kw["folds"], "methods": FairViewComparisonTests.fit1}
        bundle = {"rows": rows, "folds": kw["folds"]}
        self.assertEqual(audit.audit_support_and_methods(fitted, bundle), 7)
        broken = copy.deepcopy(fitted)
        broken["methods"]["linear_probe"]["selected"]["oof_macro_auprc"] = -1
        with self.assertRaisesRegex(ValueError, "argmax"):
            audit.audit_support_and_methods(broken, bundle)


if __name__ == "__main__":
    unittest.main()
