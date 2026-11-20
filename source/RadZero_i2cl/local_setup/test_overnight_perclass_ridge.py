"""CPU-only group isolation, class isolation and binary equivalence checks."""

import json
from pathlib import Path
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from overnight_linear_residual import FrozenCache
from overnight_cache_methods import normalized_features
from overnight_perclass_ridge import (
    PENALTIES, STRENGTHS, check_binary_selection, class_candidate_order,
    fit_selected, predict_selected, support_search,
)
from overnight_perclass_ridge_control import PerClassController
from overnight_ridge_wide_control import parse_args as control_parse_args

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "local_assets/outputs/overnight_20261020_2150/features"
WIDE = PROJECT / "local_assets/outputs/overnight_v2_20261020_2250/ridge_wide"


class PerClassRidgeTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]])
        self.groups = ["a", "a", "b", "b", "c", "d"]
        rng = np.random.RandomState(42)
        self.features = normalized_features(rng.normal(size=(6, 2, 7)), 2)
        self.base = rng.normal(size=(6, 2))

    def search(self, labels=None, groups=None):
        return support_search(
            self.features, self.labels if labels is None else labels,
            self.base, self.groups if groups is None else groups, ["a", "b"],
            [.1, 1.], [0., .3, 1.])

    def test_all_group_related_images_are_excluded_and_fold_weights_fit_only(self):
        records, audits, choices, assembled, _ = self.search()
        self.assertEqual(len(records), 6)
        self.assertEqual(assembled.shape, self.base.shape)
        for audit in audits.values():
            for fold in audit["folds"]:
                fit, held = fold["fit_indices"], fold["held_indices"]
                self.assertFalse(set(self.groups[i] for i in fit)
                                 & set(self.groups[i] for i in held))
                self.assertEqual(fold["fit_positive_counts"],
                                 self.labels[fit].sum(axis=0).tolist())

    def test_another_class_labels_do_not_change_this_class_choice_or_weights(self):
        _, _, original, _, _ = self.search()
        changed_labels = self.labels.copy()
        changed_labels[:, 1] = 1 - changed_labels[:, 1]
        _, _, changed, _, _ = self.search(labels=changed_labels)
        self.assertEqual(original[0], changed[0])
        first, _ = fit_selected(self.features, self.labels, self.base, original)
        second, _ = fit_selected(self.features, changed_labels, self.base, changed)
        np.testing.assert_array_equal(first[0], second[0])

    def test_unidentifiable_fold_states_fall_back_to_baseline(self):
        labels = np.asarray([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
        groups = ["positive"] * 3 + ["negative"] * 3
        _, _, choices, assembled, _ = self.search(labels, groups)
        self.assertTrue(all(row["zero_state_fallback"] for row in choices))
        self.assertTrue(all(row["strength"] == 0 for row in choices))
        directions, strengths = fit_selected(self.features, labels, self.base, choices)
        np.testing.assert_array_equal(predict_selected(self.features, self.base, directions, strengths),
                                      self.base)
        np.testing.assert_array_equal(assembled, self.base)

    def test_exact_tie_prefers_smaller_strength_then_stronger_regularization(self):
        def row(strength, penalty):
            return {"selection_score": .9, "oof_bce": {"weighted_bce": .3},
                    "strength": strength, "parameter": penalty, "candidate_id": str(penalty)}
        self.assertLess(class_candidate_order(row(.3, .1)), class_candidate_order(row(1., 100.)))
        self.assertLess(class_candidate_order(row(.3, 100.)), class_candidate_order(row(.3, .1)))

    def test_supervisor_uses_only_new_output_and_enforces_binary_reference(self):
        args = control_parse_args([
            "--raw-root", "/tmp/perclass_new", "--state-dir", "/tmp/perclass_campaign/cpu",
            "--gpu-state", "/tmp/perclass_campaign/controller/state.json"])
        controller = PerClassController(args)
        self.assertEqual(len(controller.state["jobs"]), 5)
        self.assertEqual(args.max_parallel, 2)
        for job in controller.state["jobs"].values():
            self.assertEqual(Path(job["output"]).parent, Path("/tmp/perclass_new/ridge_perclass"))
            argv = controller.command_for_job(job)
            self.assertEqual(argv[-2:], ["--binary-reference-root", "/tmp/perclass_new/ridge_wide"])

    def test_real_binary_datasets_match_global_support_choices_and_all_predictions(self):
        if not all((WIDE / name / "selection.json").exists() for name in ("SIIM", "RSNA")):
            self.skipTest("Local global-wide references not available")
        with threadpool_limits(limits=4):
            for dataset in ("SIIM", "RSNA"):
                with self.subTest(dataset=dataset):
                    cache = FrozenCache(OLD / dataset)
                    support = cache.support()
                    labels = support["labels"]
                    base = np.asarray(cache.array("support", "baseline_logits"), dtype=np.float64)
                    features = normalized_features(cache.array("support", "class_local"), 1)
                    _, _, choices, _, _ = support_search(
                        features, labels, base, support["groups"], support["class_names"],
                        PENALTIES, STRENGTHS)
                    check = check_binary_selection(WIDE, dataset, choices[0])
                    directions, strengths = fit_selected(features, labels, base, choices)
                    test_base = np.asarray(cache.array("test", "baseline_logits"), dtype=np.float64)
                    test_features = cache.array("test", "class_local")
                    expected = np.load(WIDE / dataset / check["candidate_id"] / "predictions.npy",
                                       allow_pickle=False)
                    for start in range(0, len(expected), 512):
                        stop = min(start + 512, len(expected))
                        actual = predict_selected(
                            normalized_features(test_features[start:stop], 1), test_base[start:stop],
                            directions, strengths)
                        np.testing.assert_array_equal(actual, expected[start:stop])


if __name__ == "__main__":
    unittest.main()
