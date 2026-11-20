"""Checks of rare-class macro validity and paired spatial cluster statistics."""
import unittest
from types import SimpleNamespace

import numpy as np

from summarize_stability import (BRANCHES, neutralize_group_language, spatial_bootstrap,
                                 strict_macro, make_patient_draw_plan, aggregate_classification)


class StabilityStatisticsTests(unittest.TestCase):
    def test_five_seed_mean_averages_metrics_and_preserves_paired_draws(self):
        results = []
        offsets = np.asarray([-.04, -.02, 0, .02, .04])
        for offset in offsets:
            point = np.full((4, 2, 3), .5)
            point[1] += .10 + offset
            point[2] += .01
            point[3] += .15 + offset
            # Both baseline and adapted receive the same random draw effect.
            draws = point[None] + np.asarray([-.05, 0, .05])[:, None, None, None]
            results.append(SimpleNamespace(point_metrics=point, replicate_metrics=draws,
                                           task_names=("A", "B")))
        details, rows, by_class = aggregate_classification("M", "D", results, "patient")
        result = details["tta_center_refined"]["macro"]["auroc"]
        self.assertAlmostEqual(result["mean_percent"], 65)
        self.assertAlmostEqual(result["support_seed_sd_pp"], 100 * offsets.std(ddof=1))
        np.testing.assert_allclose(result["mean_gain_paired_cluster_bootstrap"]["ci"], [15, 15])
        self.assertEqual(rows[-1]["both_primary_positive_seeds"], 5)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(by_class), 8)

    def test_macro_does_not_drop_an_invalid_rare_class(self):
        points = np.asarray([[[.7, .4], [.8, .5]]])
        draws = np.asarray([[[[.6, .3], [np.nan, np.nan]]],
                            [[[.8, .5], [.9, .6]]]])
        point, sample = strict_macro(points, draws)
        np.testing.assert_allclose(point, [[.75, .45]])
        self.assertTrue(np.isnan(sample[0]).all())
        np.testing.assert_allclose(sample[1], [[.85, .55]])

    def test_spatial_macro_equals_explicit_repeated_group_samples(self):
        # Two images belong to one group; a fifth classification-only group has
        # no localisation annotation. Do not average image-level hit fractions.
        groups = np.asarray(["g1", "g1", "g2", "g3", "g4", "g5"])
        expected = {"image_paths": np.asarray(["a", "b", "c", "d", "e", "f"]),
                    "group_ids": groups, "class_names": np.asarray(["A", "B"])}
        plan = make_patient_draw_plan(groups, n_resamples=51, seed=8)
        cases = np.asarray([[1, 0], [1, 1], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
        hits = np.asarray([[1, 0], [0, 1], [0, 0], [1, 1], [0, 0]], dtype=np.uint8)
        data = {"image_paths": expected["image_paths"][:5], "group_ids": groups[:5],
                "class_names": expected["class_names"]}
        for i, branch in enumerate(BRANCHES):
            data[branch + "_cases"] = cases
            data[branch + "_hits"] = hits if i % 2 == 0 else cases
        points, draws, summary = spatial_bootstrap(data, expected, plan, "image_exact_duplicate_group")
        np.testing.assert_allclose(points[0], [.5, 2 / 3])
        self.assertEqual(summary["resampling_unit"], "image_exact_duplicate_group")
        self.assertFalse(summary["patient_identity_verified"])
        for r, counts in enumerate(plan.counts):
            expanded = np.concatenate([np.tile(np.flatnonzero(groups[:5] == group), int(count))
                                       for group, count in zip(plan.patient_ids, counts)])
            for b, branch in enumerate(BRANCHES):
                denominator = cases[expanded].sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    target = data[branch + "_hits"][expanded].sum(axis=0) / denominator
                np.testing.assert_allclose(draws[r, b], target, equal_nan=True)
        invalid = np.isnan(draws[:, 0]).any(axis=1)
        self.assertTrue(invalid.any())
        self.assertEqual(summary["branches"][BRANCHES[0]]["macro_pg"]["invalid_replicates"], int(invalid.sum()))

    def test_nonpatient_summary_language_is_not_misrepresented(self):
        summary = {"patients": 3, "resampling_unit": "patient", "test_counts": {
            "A": {"patients_with_positive_image": 2}}, "scope": "test-patient sampling"}
        result = neutralize_group_language(summary)
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["test_counts"]["A"]["groups_with_positive_image"], 2)
        self.assertNotIn("patient", repr(result))

    def test_spatial_changing_case_masks_is_rejected(self):
        expected = {"image_paths": np.asarray(["a", "b"]), "group_ids": np.asarray(["a", "b"]),
                    "class_names": np.asarray(["A"])}
        data = dict(expected)
        for branch in BRANCHES:
            data[branch + "_hits"] = np.asarray([[1], [0]])
            data[branch + "_cases"] = np.asarray([[1], [1]])
        data[BRANCHES[1] + "_cases"] = np.asarray([[1], [0]])
        plan = make_patient_draw_plan(expected["group_ids"], n_resamples=3)
        with self.assertRaisesRegex(ValueError, "case mask changed"):
            spatial_bootstrap(data, expected, plan, "image_exact_duplicate_group")


if __name__ == "__main__":
    unittest.main()
