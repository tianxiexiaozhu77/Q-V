"""CPU tests for threshold freezing, paired patient statistics and access gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.qata_multiseed_20261024 import common
from experiments.qata_multiseed_20261024 import evaluate_fixed_test as evaluation
from experiments.qata_multiseed_20261024 import select_thresholds as selection


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_threshold_tie_uses_smallest_grid_value(self):
        curves = np.zeros((3, 101), dtype=np.float32)
        curves[:, 61] = [.4, .7, .4]
        curves[:, 62] = curves[:, 61]
        result = selection.choose_threshold(curves)
        self.assertEqual(result["threshold_index"], 61)
        self.assertEqual(result["threshold"], selection.metrics.THRESHOLDS[61])
        self.assertAlmostEqual(result["validation_dice"], .5)

    def test_invalid_threshold_curves_fail(self):
        for curves in (np.zeros((0, 101), np.float32), np.zeros((3, 100), np.float32),
                       np.full((3, 101), np.nan, np.float32), np.full((3, 101), 1.1, np.float32)):
            with self.assertRaises(ValueError):
                selection.choose_threshold(curves)

    def test_patient_bootstrap_matches_explicit_image_resampling(self):
        # Patient A contributes three images and B/C one each. A patient-mean
        # shortcut would change this image-equal estimator and is caught here.
        patients = ["sub-SA", "sub-SA", "sub-SA", "sub-SB", "sub-SC"]
        scores = np.random.default_rng(13).uniform(.1, .85, (5, 5, 5)).astype(np.float32)
        branches, pairs, boot, draws, unique, per_seed = evaluation.summarize_scores(scores, patients, 127, 27)
        manual = []
        ids = np.asarray(patients)
        for selected_patients in draws:
            indices = np.concatenate([np.flatnonzero(ids == unique[index]) for index in selected_patients])
            manual.append(scores[indices].astype(np.float64).mean(axis=0))
        np.testing.assert_allclose(boot, manual, rtol=0, atol=1e-15)
        q, p = common.BRANCHES.index("query_refined"), common.BRANCHES.index("positive_query")
        manual_delta = boot[:, :, q].mean(axis=1) - boot[:, :, p].mean(axis=1)
        np.testing.assert_allclose(pairs["Q_minus_Positive"]["delta_ci95"], np.quantile(manual_delta, [.025, .975]))
        np.testing.assert_allclose(branches["query_refined"]["mean_test_dice_ci95"],
                                   np.quantile(boot[:, :, q].mean(axis=1), [.025, .975]))
        expected_sd = per_seed[:, q].std(ddof=1)
        self.assertEqual(branches["query_refined"]["support_seed_sd"], expected_sd)
        self.assertNotEqual(expected_sd, per_seed[:, q].std(ddof=0))

    def test_support_seed_variability_is_not_bootstrapped_as_patients(self):
        scores = np.full((4, 5, 5), .25, dtype=np.float32)
        # Each seed's Q gain is constant across patients, but varies by seed.
        scores[:, :, 2] += np.asarray([0., .0625, .125, .1875, .25], np.float32)
        branches, pairs, *_ = evaluation.summarize_scores(scores, ["sub-SA", "sub-SA", "sub-SB", "sub-SC"], 100, 9)
        result = pairs["Q_minus_Original"]
        self.assertGreater(branches["query_refined"]["support_seed_sd"], 0)
        np.testing.assert_array_equal(result["delta_ci95"], [.125, .125])
        self.assertEqual(result["mean_delta_dice"], .125)
        self.assertEqual(result["positive_seed_count"], 4)
        self.assertEqual(result["role"], "primary")
        self.assertEqual(pairs["Q_minus_Positive"]["role"], "primary")
        self.assertEqual(sum(value["role"] == "primary" for value in pairs.values()), 2)

    def test_atomic_record_never_overwrites_frozen_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            selection.publish_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                selection.publish_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_statistics_cannot_change_frozen_comparisons_or_resampling(self):
        config = {
            "primary_contrasts": [[left, right] for _, left, right, role in evaluation.COMPARISONS if role == "primary"],
            "secondary_contrasts": [[left, right] for _, left, right, role in evaluation.COMPARISONS if role == "secondary"],
            "bootstrap": {"replicates": 2000, "seed": 20261024, "unit": "patient", "support_standard_deviation_ddof": 1},
        }
        evaluation.verify_statistics_protocol(config, 2000, 20261024)
        with self.assertRaisesRegex(ValueError, "Bootstrap options"):
            evaluation.verify_statistics_protocol(config, 1000, 20261024)
        with self.assertRaisesRegex(ValueError, "Bootstrap options"):
            evaluation.verify_statistics_protocol(config, 2000, 42)
        config["primary_contrasts"].pop()
        with self.assertRaisesRegex(ValueError, "Paired contrasts"):
            evaluation.verify_statistics_protocol(config, 2000, 20261024)

    def test_bad_threshold_file_cannot_open_test_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            threshold = root / "thresholds.json"
            threshold.write_text(json.dumps({"status": "not_frozen"}))
            with (patch.object(evaluation, "read_protocol", return_value=(protocol, {})),
                  patch.object(evaluation, "load_rows") as rows,
                  patch.object(evaluation.metrics, "load_positive_masks") as masks,
                  patch.object(evaluation, "load_maps") as maps):
                with self.assertRaisesRegex(ValueError, "frozen threshold"):
                    evaluation.evaluate(protocol, root / "maps", threshold, root / "evaluation")
                rows.assert_not_called()
                maps.assert_not_called()
                masks.assert_not_called()

    def test_all_test_map_identities_checked_before_any_test_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            threshold = root / "thresholds.json"
            threshold.write_text("{}")
            shared = {"manifest": {"path": "manifest", "sha256": "abc"},
                      "query_fit": {"path": "fit", "sha256": "def"}}
            frozen = {"validation_predictions": {str(seed): shared for seed in common.SEEDS}}
            calls = []

            def fake_rows(path, config, split):
                return [{"image": split + ".png", "patient_id": "sub-S" + split}]

            def failed_maps(*args):
                calls.append(args[-1])
                if args[-1] == common.SEEDS[-1]:
                    raise ValueError("native map identity mismatch")
                return {}, shared

            with (patch.object(evaluation, "read_protocol", return_value=(protocol, {})),
                  patch.object(evaluation, "verify_frozen_thresholds", return_value=frozen),
                  patch.object(evaluation, "verify_statistics_protocol"),
                  patch.object(evaluation, "split_descriptor", return_value=(root / "test.json", "abc")),
                  patch.object(evaluation, "load_rows", side_effect=fake_rows),
                  patch.object(evaluation, "load_maps", side_effect=failed_maps),
                  patch.object(evaluation.metrics, "load_positive_masks") as masks):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    evaluation.evaluate(protocol, root / "maps", threshold, root / "evaluation")
                self.assertEqual(calls, list(common.SEEDS))
                masks.assert_not_called()

    def test_complete_freeze_validation_only_and_mutation_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            validation_path = root / "validation.json"
            rows = [{"image": str(root / "a.png"), "patient_id": "sub-SA"},
                    {"image": str(root / "b.png"), "patient_id": "sub-SB"}]
            validation_path.write_text(json.dumps(rows))
            prediction = root / "validation_maps.npz"
            prediction.write_bytes(b"synthetic unit-test provenance, not inference")
            proof = {"path": str(prediction), "sha256": selection.metrics.sha256(prediction)}
            requested_splits = []

            def checked_rows(path, config, split):
                requested_splits.append(split)
                self.assertEqual(split, "validation")
                return rows

            curves = np.zeros((2, 101), np.float32)
            curves[:, 63] = [.5, .75]
            output = root / "thresholds.json"
            with (patch.object(selection, "read_protocol", return_value=(protocol, {})),
                  patch.object(selection, "load_rows", side_effect=checked_rows),
                  patch.object(selection, "split_descriptor", return_value=(validation_path, None)),
                  patch.object(selection, "load_maps", return_value=({branch: None for branch in common.BRANCHES}, proof)),
                  patch.object(selection.metrics, "load_positive_masks", return_value=(None, {})),
                  patch.object(selection.metrics, "score_maps", return_value=curves)):
                result = selection.select_thresholds(protocol, root, output)
            self.assertEqual(requested_splits, ["validation"])
            self.assertEqual(sum(len(items) for items in result["thresholds"].values()), 25)
            with (patch.object(evaluation, "split_descriptor", return_value=(validation_path, None)),
                  patch.object(evaluation, "load_rows", return_value=rows)):
                evaluation.verify_frozen_thresholds(output, protocol, {})
                # Reject a threshold altered after the full frozen record exists,
                # even if it remains a valid value on the threshold grid.
                changed = json.loads(output.read_text())
                changed["thresholds"]["42"]["query_refined"]["threshold_index"] = 64
                changed["thresholds"]["42"]["query_refined"]["threshold"] = float(selection.metrics.THRESHOLDS[64])
                output.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "validation optimum"):
                    evaluation.verify_frozen_thresholds(output, protocol, {})
                output.write_text(json.dumps(result))
                prediction.write_bytes(b"changed artifact")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    evaluation.verify_frozen_thresholds(output, protocol, {})

    def test_smoke_map_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "smoke": 16}))
            with self.assertRaisesRegex(ValueError, "smoke"):
                common.load_maps(root, root / "protocol.json", {}, "validation", 42)


if __name__ == "__main__":
    unittest.main()
