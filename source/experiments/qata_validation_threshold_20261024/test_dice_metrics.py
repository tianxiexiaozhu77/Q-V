"""Numerical/protocol checks independent of inference; run in AlphaRAD environment."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore

from dice_metrics import (BRANCHES, THRESHOLDS, author_mean, decode_relative_rle,
                         dice_from_probabilities, load_positive_masks, load_rows,
                         patient_bootstrap, probabilities, score_maps, sha256)
from evaluate_fixed_test import verify_frozen_thresholds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class DiceNumerics(unittest.TestCase):
    def test_strict_threshold_ties_match_torchmetrics(self):
        prob = np.asarray([[[0., .5, .64], [.64, .7, 1.]],
                           [[.3, .64, .64], [.5, 1., .9]]], dtype=np.float32)
        truth = np.asarray([[[0, 1, 0], [1, 1, 0]], [[1, 0, 1], [0, 0, 1]]], dtype=bool)
        thresholds = np.array([0., .5, .64, .7, 1.])
        fast = dice_from_probabilities(prob, truth, thresholds)
        for j, t in enumerate(thresholds):
            for i in range(len(prob)):
                metric = DiceScore(num_classes=2, include_background=False,
                                   average="macro", aggregation_level="samplewise")
                pred = (torch.tensor(prob[i:i+1, None]) > float(t)).long()
                target = torch.tensor(truth[i:i+1, None]).long()
                self.assertEqual(float(fast[i, j]), float(metric(pred, target)))

    def test_cls_included_in_minmax_before_spatial_removal(self):
        attention = np.asarray([[[100., 0., 1., 2., 3.]]], dtype=np.float32)
        # A 5-token test map is not one of the author CLS grids: use 197 tokens.
        attention = np.arange(197, dtype=np.float32)[None, None]
        attention[0, 0, 0] = 1000
        probability = probabilities(attention, (14, 14))
        expected = torch.sigmoid(torch.tensor((attention[0, 0, 1:] - 1) / 999)).reshape(14, 14).numpy()
        np.testing.assert_array_equal(probability[0], expected)
        self.assertLess(float(probability.max()), .55)

    def test_empty_masks_rejected(self):
        with self.assertRaises(ValueError):
            dice_from_probabilities(np.ones((1, 2, 2)), np.zeros((1, 2, 2)), np.array([.5]))

    def test_relative_rle_includes_column_major_and_zero_start(self):
        result = decode_relative_rle("0 2 2 1", (2, 3))
        np.testing.assert_array_equal(result, np.array([[1, 0, 1], [1, 0, 0]], dtype=bool))

    def test_cluster_bootstrap_keeps_images_and_common_draws(self):
        scores = np.array([[.1, .2], [.3, .4], [.9, .7], [.5, .6]])
        patient_ids = ["sub-S1", "sub-S1", "sub-S2", "sub-S3"]
        values, draws, patients = patient_bootstrap(scores, patient_ids, replicates=33, seed=12)
        for i, sample in enumerate(draws):
            indices = [j for p in sample for j, name in enumerate(patient_ids) if name == patients[p]]
            np.testing.assert_allclose(values[i], scores[indices].mean(axis=0), atol=1e-15, rtol=0)
        self.assertTrue(np.any(np.abs(values[:, 0] - .5) > .01))

    def test_frozen_protocol_gate_rejects_modified_validation_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            protocol = directory / "protocol.json"
            protocol.write_text("{}")
            artifact = directory / "validation.npz"
            artifact.write_bytes(b"frozen validation artifact")
            record = {"status": "frozen_before_fixed_test_scoring",
                      "test_labels_used_for_selection": False, "test_predictions_used_for_selection": False,
                      "protocol": {"sha256": sha256(protocol)},
                      "code_sha256": {name: sha256(HERE / name) for name in ("dice_metrics.py", "select_thresholds.py")},
                      "branches": {b: {"threshold": .64, "threshold_index": 0} for b in BRANCHES},
                      "threshold_grid": [.64]}
            for name in ("validation_rows", "validation_predictions", "validation_curves"):
                record[name] = {"path": str(artifact), "sha256": sha256(artifact)}
            threshold = directory / "thresholds.json"
            threshold.write_text(json.dumps(record))
            verify_frozen_thresholds(threshold, protocol)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_frozen_thresholds(threshold, protocol)

    def test_against_unmodified_author_helper_on_real_validation_masks(self):
        rows_path = HERE / "protocol_v1/validation_rows.json"
        if not rows_path.exists():
            self.skipTest("Frozen validation protocol is not available yet")
        rows = load_rows(rows_path)[:3]
        masks, _ = load_positive_masks(rows)
        rng = np.random.default_rng(93)
        attention = rng.normal(size=(len(rows), 1, 1370)).astype(np.float32)
        fast = author_mean(score_maps(attention, masks, THRESHOLDS))
        source = ROOT / "AlphaRAD/cxrclip/evaluator_utils.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gather_seg_statistics")
        namespace = dict(np=np, torch=torch, F=F, cv2=cv2, DiceScore=DiceScore,
                         tqdm=lambda values, **kwargs: values)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = namespace["gather_seg_statistics"](
                attention, [r["image"] for r in rows],
                [torch.tensor(mask.astype(np.int64))[None] for mask in masks],
                [["covid"]] * len(rows), ["covid"])
        result = expected["covid"]
        self.assertEqual(float(THRESHOLDS[np.argmax(fast)]), result["best_threshold"])
        self.assertEqual(float(fast.max()), result["zeroshot_dice"])

    def test_select_without_accessible_test_annotations_then_fixed_score(self):
        """The test split does not exist at threshold selection time."""
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            row_sets, map_paths = {}, {}
            for split in ("validation", "test"):
                rows = []
                for i in range(3):
                    image = directory / f"{split}_{i}.png"
                    mask_path = directory / f"{split}_mask_{i}.png"
                    Image.fromarray(np.full((8, 8), i * 40, dtype=np.uint8)).save(image)
                    mask = np.zeros((8, 8), dtype=np.uint8)
                    mask[2:5, 1+i:4+i] = 255
                    Image.fromarray(mask).save(mask_path)
                    rows.append(dict(image=str(image), mask_path=str(mask_path),
                                     patient_id=f"sub-S{1 if split == 'validation' else 2}{i}",
                                     mask_sha256=sha256(mask_path), **{"class": "covid"}))
                row_sets[split] = rows
                map_path = directory / f"{split}_maps.npz"
                native = np.random.default_rng(3).normal(size=(3, 1, 1370)).astype(np.float32)
                np.savez(map_path, **{branch: native for branch in BRANCHES},
                         image_paths=np.asarray([r["image"] for r in rows]),
                         patient_ids=np.asarray([r["patient_id"] for r in rows]), class_names=np.asarray(["covid"]))
                map_paths[split] = map_path
            val_path, test_path = directory / "validation_rows.json", directory / "test_rows.json"
            val_path.write_text(json.dumps(row_sets["validation"]))
            test_json = json.dumps(row_sets["test"])
            protocol = directory / "protocol.json"
            protocol.write_text(json.dumps({"status": "frozen", "eta": .3, "crop_fraction": .8,
                "class_names": ["covid"], "evaluation_positive_only": True,
                "threshold_protocol": {"grid": THRESHOLDS.tolist()},
                "validation": {"rows_path": str(val_path), "rows_sha256": sha256(val_path)},
                "test": {"rows_path": str(test_path), "rows_sha256": hashlib.sha256(test_json.encode()).hexdigest()}}))
            thresholds = directory / "thresholds.json"
            command = [sys.executable, str(HERE / "select_thresholds.py"), "--protocol", str(protocol),
                       "--maps", str(map_paths["validation"]), "--output", str(thresholds)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertFalse(test_path.exists())
            frozen_sha = sha256(thresholds)
            test_path.write_text(test_json)
            output = directory / "evaluation"
            command = [sys.executable, str(HERE / "evaluate_fixed_test.py"), "--protocol", str(protocol),
                       "--thresholds", str(thresholds), "--maps", str(map_paths["test"]),
                       "--output", str(output), "--replicates", "7"]
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertEqual(sha256(thresholds), frozen_sha)
            results = json.loads((output / "RESULTS.json").read_text())
            self.assertFalse(results["test_threshold_search_performed"])
            for branch in BRANCHES:
                self.assertEqual(results["branches"][branch]["gain_over_original"], 0.)
                self.assertEqual(results["branches"][branch]["gain_over_original_ci95"], [0., 0.])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
