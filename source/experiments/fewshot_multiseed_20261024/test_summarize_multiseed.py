"""Statistical regression tests with independently calculated expected values."""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_multiseed as summary
from cluster_bootstrap import PatientDrawPlan, binary_weighted_metrics, paired_cluster_bootstrap


def sklearn_metrics(y, score):
    precision, recall, _ = precision_recall_curve(y, score)
    return np.asarray([roc_auc_score(y, score), auc(recall, precision), average_precision_score(y, score)])


class StatisticalTests(unittest.TestCase):
    def test_patient_draw_repeats_every_image_and_matches_explicit_expansion(self):
        # Unequal images per patient is deliberate. Replacing each patient with
        # one average prediction would produce a different estimand.
        groups = np.asarray(["a", "a", "a", "b", "c", "c"])
        labels = np.asarray([1, 0, 1, 0, 1, 0])
        scores = np.asarray([.9, .7, .7, .7, .2, .1])  # score ties cross labels
        counts = np.asarray([[2, 0, 1], [0, 1, 2], [1, 1, 1]], dtype=np.uint32)
        plan = PatientDrawPlan(np.asarray(["a", "b", "c"]), counts, 20261024)
        result = paired_cluster_bootstrap(labels[:, None],
            {"original": scores[:, None], "query": scores[:, None]}, groups, plan=plan)
        for replicate, draw in enumerate(counts):
            weights = draw[np.searchsorted(plan.patient_ids, groups)]
            expanded = np.repeat(np.arange(len(labels)), weights)
            target = sklearn_metrics(labels[expanded], scores[expanded])
            assert_allclose(result.replicate_metrics[replicate, 0, 0], target, atol=1e-14)
            assert_allclose(binary_weighted_metrics(labels, scores, weights), target, atol=1e-14)
        assert_allclose(result.replicate_metrics[:, 1] - result.replicate_metrics[:, 0], 0)

    def test_mean_gain_ci_uses_same_draw_and_average_of_seed_differences(self):
        points = np.asarray([[.30, .34], [.70, .72], [.40, .39], [.80, .81], [.50, .49]])
        common_noise = np.asarray([-.20, -.08, .01, .05, .13, .21])
        draws = np.repeat(points[:, None, :], len(common_noise), axis=1)
        draws += common_noise[None, :, None]
        changes = np.asarray([-.03, -.01, .00, .02, .03, .04])
        draws[:, :, 1] += np.arange(1, 6)[:, None] * changes[None, :]
        item = summary.paired_mean_summary(points, draws, 1, 0)
        expected_draws = 100 * (draws[:, :, 1] - draws[:, :, 0]).mean(axis=0)
        assert_allclose(item["mean_gain_paired_cluster_bootstrap"]["ci"],
                        np.quantile(expected_draws, [.025, .975]))
        self.assertAlmostEqual(item["mean_gain_pp"], 100 * np.mean(points[:, 1] - points[:, 0]))
        self.assertEqual((item["wins"], item["ties"], item["losses"]), (3, 0, 2))
        # Marginal interval subtraction does not preserve the shared noise.
        wrong_low = np.quantile(draws[:, :, 1].mean(0), .025) - np.quantile(draws[:, :, 0].mean(0), .975)
        self.assertGreater(abs(item["mean_gain_paired_cluster_bootstrap"]["ci"][0] - 100 * wrong_low), 10)

    def test_five_seed_metric_mean_is_not_metric_of_averaged_predictions(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = [np.asarray([0., 0., 10., 10.])] * 3 + [np.asarray([1., 1., 0., 0.])] * 2
        methods = ("original", "query", "query_view")
        fitted = [paired_cluster_bootstrap(labels[:, None],
            {"original": np.asarray([0., 1., 1., 0.])[:, None], "query": score[:, None],
             "query_view": score[:, None]}, ["a", "b", "c", "d"], n_resamples=64, seed=77)
            for score in scores]
        output = summary.summarize_arrays("m", "d", np.stack([f.point_metrics for f in fitted]),
            np.stack([f.replicate_metrics for f in fitted]), ["task"], "patient", methods)
        row = next(row for row in output["means"] if row["method"] == "query")
        self.assertAlmostEqual(row["auroc_mean_percent"], 60.)
        self.assertAlmostEqual(row["auroc_sd_pp"], 100 * np.std([1, 1, 1, 0, 0], ddof=1))
        self.assertEqual(roc_auc_score(labels, np.stack(scores).mean(0)), 1.)
        pair = next(row for row in output["comparisons"]
                    if row["target"] == "query" and row["reference"] == "query_view")
        assert_allclose([pair["auroc_delta_pp"], pair["auroc_ci_low_pp"], pair["auroc_ci_high_pp"]], 0.)
        self.assertEqual(pair["auroc_ties"], 5)
        for f in fitted[1:]:
            self.assertEqual(f.plan.sha256, fitted[0].plan.sha256)

    def test_macro_does_not_silently_drop_invalid_class(self):
        methods = ("original", "query", "query_view")
        points = np.full((5, 3, 2, 3), .5)
        draws = np.full((5, 4, 3, 2, 3), .5)
        draws[:, 1, :, 1, :] = np.nan
        output = summary.summarize_arrays("m", "d", points, draws, ["a", "b"], "patient", methods)
        for row in output["means"]:
            self.assertEqual(row["auroc_valid_replicates"], 3)
            self.assertEqual(row["auroc_invalid_replicates"], 1)
            self.assertEqual(row["auroc_valid_fraction"], .75)

    def test_spatial_pg_is_class_macro_with_classification_group_weights(self):
        expected = {"image_paths": np.asarray(["a", "b", "c", "without_box"]),
                    "group_ids": np.asarray(["g1", "g1", "g2", "g3"]),
                    "class_names": np.asarray(["x", "y"])}
        counts = np.asarray([[2, 1, 0], [1, 0, 2], [0, 1, 2]], dtype=np.uint32)
        plan = PatientDrawPlan(np.asarray(["g1", "g2", "g3"]), counts, 20261024)
        data = {"image_paths": expected["image_paths"][:3], "group_ids": expected["group_ids"][:3],
                "class_names": expected["class_names"]}
        cases = np.asarray([[1, 1], [1, 0], [1, 0]])
        hits = np.asarray([[1, 1], [0, 0], [0, 0]])
        for method in summary.NATIVE_METHODS:
            data[method + "_cases"], data[method + "_hits"] = cases, hits
        native = {"spatial": {method: {"pointing_game": (1 / 3 + 1) / 2}
                              for method in summary.NATIVE_METHODS}}
        points, draws, metadata = summary.spatial_bootstrap(data, expected, plan,
            [{"image": path} for path in data["image_paths"]], native)
        assert_allclose(points[0, :, 0], [1 / 3, 1])
        assert_allclose(draws[0, 0, :, 0], [2 / 5, 1])
        assert_allclose(draws[1, 0, :, 0], [1 / 2, 1])
        self.assertTrue(np.isnan(draws[2, 0, 1, 0]))
        self.assertAlmostEqual(points[0].mean(), 2 / 3)
        self.assertNotAlmostEqual(points[0].mean(), hits.sum() / cases.sum())
        self.assertEqual(metadata["draw_plan_sha256"], plan.sha256)

    def test_all_predeclared_competitors_are_present_without_test_selection(self):
        pairs = summary.comparison_pairs()
        self.assertEqual(len(pairs), 28)
        for method in summary.METHODS[1:]:
            self.assertIn((method, "original"), pairs)
        for target in ("query", "query_view"):
            for reference in summary.METHODS:
                if reference != target:
                    self.assertIn((target, reference), pairs)


if __name__ == "__main__":
    unittest.main()
