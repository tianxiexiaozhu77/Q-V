"""Tests for pairing, macro estimands, spatial geometry, and endpoint identity."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev
from cluster_bootstrap import PatientDrawPlan


class EvaluationTests(unittest.TestCase):
    def test_paired_factorial_interaction_cancels_shared_test_noise(self):
        points = np.zeros((5, 5, 2, 1))
        points[:, :, :, 0] = np.asarray([.5, .55, .58, .65, .6])[None, :, None]
        shared = np.asarray([-.1, -.03, 0., .05, .1])
        draws = points[:, None] + shared[None, :, None, None, None]
        result = ev.interaction_summary("toy", points, draws, ("auroc",), "patient")
        assert_allclose(result["auroc_delta_pp"], 2., atol=1e-12)
        assert_allclose([result["auroc_ci_low_pp"], result["auroc_ci_high_pp"]], [2., 2.], atol=1e-12)

    def test_spatial_macro_resamples_all_patient_images_and_keeps_missing_class_invalid(self):
        groups = np.asarray(["p1", "p1", "p2"])
        counts = np.asarray([[2, 1, 0], [1, 0, 2], [0, 1, 2]], dtype=np.uint32)
        plan = PatientDrawPlan(np.asarray(["p1", "p2", "p3"]), counts, 42)
        cases = np.asarray([[1, 1], [1, 0], [1, 0]])
        values = np.asarray([[1., .6], [0., 0.], [0., 0.]])
        points, draws = ev.spatial_group_bootstrap(values, cases, groups, plan)
        assert_allclose(points, [1 / 3, .6])
        assert_allclose(draws[0], [2 / 5, .6])
        assert_allclose(draws[1], [1 / 2, .6])
        self.assertTrue(np.isnan(draws[2, 1]))
        self.assertTrue(np.isnan(draws.mean(axis=1)[2]))
        with self.assertRaises(ValueError):
            ev.spatial_group_bootstrap(values, cases, np.asarray(["p1", "p1", "zz"]), plan)

    def test_map_selection_reorders_fixed_cohort_and_enforces_readout_invariance(self):
        expected = {"image_paths": np.asarray(["a", "b", "c"]),
                    "class_names": np.asarray(["d"]), "group_ids": np.asarray(["g1", "g2", "g3"])}
        data = {**expected, "includes_cls": np.asarray(True), "grid_size": np.asarray(37)}
        data.update({branch: np.zeros((3, 1, 1370), dtype=np.float32) for branch in ev.NATIVE})
        data["query"][:] = 1
        data["aggregation_only"][:] = 1
        selected, equality = ev.select_spatial_maps(data, expected, [{"image": "c"}, {"image": "a"}])
        self.assertEqual(selected["group_ids"].tolist(), ["g3", "g1"])
        self.assertTrue(all(v == 0 for v in equality.values()))
        data["readout_only"][0, 0, 0] = 1
        with self.assertRaises(AssertionError):
            ev.select_spatial_maps(data, expected, [{"image": "a"}])

    def test_constant_square_map_energy_is_box_union_fraction_and_cls_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "image.png")
            Image.new("L", (4, 4)).save(path)
            maps = np.zeros((1, 1, 1370), dtype=np.float32)
            maps[..., 0] = 10000  # CLS must never enter image-space softmax.
            rows = [{"image": path, "boxes": [[0, 0, 1, 1], [1, 1, 2, 2]],
                     "class_names_present": ["d", "d"]}]
            energy, cases = ev.bbox_softmax_energy(maps, rows, ["d"])
            assert_allclose(energy, [[7 / 16]], atol=1e-8)
            assert_allclose(cases, [[1]])

    def test_non_square_map_energy_respects_center_crop_and_off_crop_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = str(Path(temporary) / "wide.png")
            Image.new("L", (200, 100)).save(path)
            maps = np.zeros((1, 2, 1370), dtype=np.float32)
            rows = [{"image": path, "boxes": [[0, 0, 200, 100], [0, 0, 10, 100]],
                     "class_names_present": ["all", "off_crop"]}]
            energy, cases = ev.bbox_softmax_energy(maps, rows, ["all", "off_crop"])
            assert_allclose(energy, [[1., 0.]], atol=1e-6)
            assert_allclose(cases, [[1, 1]])

    def test_native_prediction_verifier_rejects_label_leak_and_reordered_images(self):
        expected = {"image_paths": np.asarray(["a", "b"]), "class_names": np.asarray(["d"]),
                    "group_ids": np.asarray(["g1", "g2"]), "labels": np.asarray([[0], [1]])}
        data = {k: v for k, v in expected.items() if k != "labels"}
        data.update({branch: np.full((2, 1), .5) for branch in ev.NATIVE})
        ev.verify_native_predictions(data, expected)
        with self.assertRaises(ValueError):
            ev.verify_native_predictions({**data, "labels": expected["labels"]}, expected)
        with self.assertRaises(ValueError):
            ev.verify_native_predictions({**data, "image_paths": data["image_paths"][::-1]}, expected)

    def test_native_pg_tied_peak_and_inverse_center_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            square, wide = str(Path(temporary) / "square.png"), str(Path(temporary) / "wide.png")
            Image.new("L", (100, 100)).save(square)
            Image.new("L", (200, 100)).save(wide)
            maps = np.zeros((2, 1, 1370), dtype=np.float32)
            # A constant map selects the first row-major pixel. In a 2:1 image,
            # native center-crop inversion places that pixel at original x=50.
            rows = [{"image": square, "boxes": [[0, 0, 0, 0]], "class_names_present": ["d"]},
                    {"image": wide, "boxes": [[50, 0, 50, 0]], "class_names_present": ["d"]}]
            hits, cases, metric = ev.native_pg_hits(maps, rows, ["d"])
            assert_allclose(hits, [[1], [1]])
            assert_allclose(cases, [[1], [1]])
            self.assertEqual(metric["pointing_game"], 1.)


if __name__ == "__main__":
    unittest.main()
