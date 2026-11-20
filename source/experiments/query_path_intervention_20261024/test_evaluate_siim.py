"""SIIM-specific tests: relative RLE, native spatial geometry and estimands."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_siim as ev


class SIIMEvaluationTests(unittest.TestCase):
    def test_relative_zero_based_column_major_runs_and_multiple_mask_union(self):
        # Flat Fortran-order foreground at indices 0,1,3,4. The second run's
        # start is previous END + 1; it is neither absolute nor one-based.
        result = ev.decode_siim_mask(["0 2 1 1", "4 1", "-1"], [2, 3])
        assert_array_equal(result, [[1, 0, 1], [1, 1, 0]])
        with self.assertRaises(ValueError):
            ev.decode_siim_mask(["5 2"], [2, 3])
        with self.assertRaises(ValueError):
            ev.decode_siim_mask(["1 2 3"], [2, 3])

    def test_pixel_ranking_and_peak_require_no_fitted_threshold(self):
        mask = np.asarray([[0, 1], [0, 1]], dtype=bool)
        good = np.asarray([[0., 1.], [0., 1.]], dtype=np.float32)
        assert_allclose(ev.pixel_metrics(good, mask), [1., 1.])
        assert_allclose(ev.pixel_metrics(-good, mask), [0., 0.])
        # A constant map produces AUC .5; row-major argmax picks background.
        assert_allclose(ev.pixel_metrics(np.zeros((2, 2)), mask), [.5, 0.])
        with self.assertRaises(ValueError):
            ev.pixel_metrics(good, np.zeros_like(mask))

    def test_positive_image_metrics_exclude_negatives_and_cls(self):
        maps = np.zeros((2, 1, 1370), dtype=np.float32)
        maps[..., 0] = 10000
        rows = [{"labels": [1], "rle": ["0 1"], "original_shape": [2, 2]},
                {"labels": [0], "rle": ["-1"], "original_shape": [2, 2]}]
        values, cases, foreground = ev.spatial_image_values(maps, rows, workers=1)
        assert_allclose(values, [[.5, 1.], [0., 0.]])
        assert_array_equal(cases, [[1], [0]])
        assert_array_equal(foreground, [1, 0])

    def test_native_grounding_equalities_enforced(self):
        expected = {"image_paths": np.asarray(["a"]), "group_ids": np.asarray(["g"]),
                    "class_names": np.asarray(["Pneumothorax"]), "labels": np.asarray([[1]])}
        data = {key: value for key, value in expected.items() if key != "labels"}
        data.update(includes_cls=np.asarray(True), grid_size=np.asarray(37))
        data.update({branch: np.zeros((1, 1, 1370), dtype=np.float32) for branch in ev.NATIVE})
        ev.native_maps(data, expected)
        data["readout_only"][0, 0, 1] = 1
        with self.assertRaises(AssertionError):
            ev.native_maps(data, expected)

    def test_actual_author_rle_functions_interoperate_on_nonsquare_masks(self):
        source_path = ev.base.MEDICAL / "AlphaRAD/cxrclip/evaluator_utils.py"
        source = ast.parse(source_path.read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in ("mask2rle", "rle2mask")]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), namespace)
        rng = np.random.default_rng(42)
        for height, width in ((2, 3), (7, 5), (32, 32)):
            mask = rng.random((height, width)) > .7
            mask[0, 0] = True
            rle = namespace["mask2rle"](mask)
            author = namespace["rle2mask"](rle, width, height)
            assert_array_equal(author, mask)
            assert_array_equal(ev.decode_siim_mask([rle], [height, width]), author)

    def test_native_receipt_rejects_changed_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = root / "catalog.json"
            protocol.write_text("{}")
            receipt = {"protocol_sha256": ev.base.sha(protocol), "test_labels_read": False,
                "test_masks_read": False, "test_rle_read": False, "frozen_model": True,
                "support_seeds": list(ev.SEEDS), "views": 1, "eta": .3,
                "artifact_sha256": {}, "seed_artifacts": {}}
            for seed in ev.SEEDS:
                folder = root / f"seed_{seed}"
                folder.mkdir()
                receipt["seed_artifacts"][str(seed)] = {}
                for name in ("predictions.npz", "native_maps.npz", "query_fit.npz", "support_manifest.json"):
                    path = folder / name
                    path.write_bytes(b"frozen-fixture")
                    digest = ev.base.sha(path)
                    receipt["seed_artifacts"][str(seed)][name] = digest
                    receipt["artifact_sha256"][str(path.relative_to(root))] = digest
            self.assertEqual(len(ev.verify_inference_receipt(receipt, root, protocol)), 20)
            (root / "seed_42/predictions.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                ev.verify_inference_receipt(receipt, root, protocol)


if __name__ == "__main__":
    unittest.main()
