"""CPU tests for exact ranking ties, fold-only BCE weights and test isolation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from overnight_cache_methods import leave_group_out, normalized_features, sha256, fingerprint
from overnight_oof_loss_tiebreak import (
    build_bce_folds, choose_candidates, parse_args, run, weighted_oof_bce,
)


class LossTiebreakTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.asarray([
            [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
        self.groups = ["a", "a", "b", "b", "c", "d"]
        self.features = normalized_features(np.asarray([
            [1., .1], [1., .2], [.1, 1.], [.2, 1.], [1., .15], [.15, 1.]]), 2)
        self.baseline = (self.labels.astype(float) * 2 - 1) * .25
        _, self.audit = leave_group_out(
            self.features, self.labels, self.baseline, self.groups, "proto", 1.)
        self.folds = build_bce_folds(self.labels, self.groups, self.audit)

    def test_weighted_oof_bce_matches_torch_raw_logit_loss(self):
        actual = weighted_oof_bce(self.baseline, self.labels, self.folds)
        total, count = 0., 0
        for fold in self.folds:
            held, valid = fold["held_indices"], np.asarray(fold["valid_classes"])
            weights = [value for value in fold["pos_weight"] if value is not None]
            z = torch.tensor(self.baseline[held][:, valid], dtype=torch.float64)
            y = torch.tensor(self.labels[held][:, valid], dtype=torch.float64)
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(
                z, y, pos_weight=torch.tensor(weights, dtype=torch.float64), reduction="sum"))
            count += y.numel()
        self.assertAlmostEqual(actual["weighted_bce"], total / count, places=13)
        self.assertEqual(actual["effective_entries"], count)
        self.assertEqual(actual["excluded_entries"], self.labels.size - count)

    def test_fold_weights_ignore_held_group_labels_and_derive_from_fit(self):
        fold = next(fold for fold in self.folds if fold["held_indices"] == [0, 1])
        self.assertEqual(fold["fit_positive_counts"], [1, 3])
        self.assertEqual(fold["pos_weight"], [3., 1. / 3])
        altered = self.labels.copy()
        altered[:2] = 1 - altered[:2]
        _, audit = leave_group_out(
            self.features, altered, self.baseline, self.groups, "proto", 1.)
        after = build_bce_folds(altered, self.groups, audit)
        same = next(item for item in after if item["held_indices"] == [0, 1])
        self.assertEqual(same["pos_weight"], fold["pos_weight"])

    def test_missing_state_classes_are_excluded_not_assigned_full_support_weights(self):
        labels = self.labels.copy()
        labels[:, 0] = [1, 1, 0, 0, 0, 0]
        _, audit = leave_group_out(
            self.features, labels, self.baseline, self.groups, "proto", 1.)
        folds = build_bce_folds(labels, self.groups, audit)
        fold = next(item for item in folds if item["held_indices"] == [0, 1])
        self.assertIsNone(fold["pos_weight"][0])
        first = weighted_oof_bce(self.baseline, labels, folds)
        changed = self.baseline.copy()
        changed[:2, 0] = -1000.
        second = weighted_oof_bce(changed, labels, folds)
        self.assertEqual(first["weighted_bce"], second["weighted_bce"])
        self.assertEqual(first["excluded_entries"], 2)

    def test_extreme_raw_logits_remain_finite_without_sigmoid_transform(self):
        logits = (self.labels * 2. - 1.) * -1000.
        result = weighted_oof_bce(logits, self.labels, self.folds)
        self.assertTrue(np.isfinite(result["weighted_bce"]))
        self.assertGreater(result["weighted_bce"], 900.)

    @staticmethod
    def record(identifier, score, bce, strength, feature="f", method="proto"):
        return {"candidate_id": identifier, "selection_score": score,
                "support_oof_bce": {"weighted_bce": bce},
                "config": {"feature": feature, "method": method, "strength": strength}}

    def test_exact_rank_ties_use_loss_then_strength(self):
        records = [
            self.record("baseline", 1., .6, 0.),
            self.record("strong", 1., .4, 3.),
            self.record("small", 1., .4, .3),
            self.record("lower_rank", np.nextafter(1., 0.), .01, .1),
        ]
        self.assertEqual(choose_candidates(records, 9)[0]["candidate_id"], "small")
        # Even a one-ULP primary advantage outranks a dramatically better BCE.
        records.append(self.record("best_rank", np.nextafter(1., 2.), 50., 1.))
        self.assertEqual(choose_candidates(records, 9)[0]["candidate_id"], "best_rank")

    def test_each_method_feature_retained_and_undefined_loss_falls_back_to_strength(self):
        records = [
            self.record("zero", 1., None, 0.),
            self.record("strong", 1., None, 3.),
            self.record("other", .9, .1, 1., method="ridge"),
        ]
        self.assertEqual([item["candidate_id"] for item in choose_candidates(records, 9)],
                         ["zero", "other"])

    def test_duplicate_held_rows_and_group_leakage_are_rejected(self):
        audit = json.loads(json.dumps(self.audit))
        audit["folds"].append(audit["folds"][0])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build_bce_folds(self.labels, self.groups, audit)
        audit = json.loads(json.dumps(self.audit))
        audit["folds"][0]["held_indices"] = [0]
        audit["folds"][0]["fit_indices"] = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, "Related"):
            build_bce_folds(self.labels, self.groups, audit)

    def test_cli_retains_zero_and_original_grid_with_top9(self):
        args = parse_args(["--cache", "/cache", "--output", "/output"])
        self.assertEqual(args.top_k, 9)
        self.assertEqual(args.strengths, [0., .1, .3, 1., 3.])
        self.assertEqual(args.features, ["class_local", "dino_cls", "aligned_cls"])
        with self.assertRaises(SystemExit):
            parse_args(["--cache", "/cache", "--output", "/output", "--strengths", ".1", "1"])

    def test_end_to_end_saturated_ranks_choose_nonzero_without_test_label_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, splits = root / "cache", root / "splits"
            cache.mkdir()
            (splits / "Toy").mkdir(parents=True)
            specs, identities, arrays = {}, {}, {}
            for split in ("support", "test"):
                prefix = "s" if split == "support" else "t"
                (splits / "Toy" / f"{split}.csv").write_text(split)
                metadata = {
                    "name": "Toy", "data_root": str(root),
                    "image_paths": [f"{prefix}{i}.png" for i in range(len(self.labels))],
                    "ids": [f"{prefix}{i}" for i in range(len(self.labels))],
                    "groups": self.groups if split == "support" else [f"t{i}" for i in range(6)],
                    "class_names": ["a", "b"], "prompts": ["a", "b"],
                }
                specs[split] = {**metadata, "labels": self.labels.copy()}
                identities[split] = fingerprint(metadata)
                (cache / f"{split}_metadata.json").write_text(json.dumps(metadata))
                arrays[split] = {"files": {}}
                values = {
                    "class_local": np.repeat(self.features[:, None], 2, axis=1).astype(np.float32),
                    "base_logits": self.baseline.astype(np.float32),
                }
                for key, value in values.items():
                    path = cache / f"{split}_{key}.npy"
                    np.save(path, value)
                    arrays[split]["files"][key] = {
                        "file": path.name, "sha256": sha256(path),
                        "shape": list(value.shape), "dtype": str(value.dtype)}
            np.save(cache / "support_labels.npy", self.labels)
            manifest = {
                "status": "complete", "dataset": "Toy", "split_dir": str(splits),
                "class_names": ["a", "b"], "splits": arrays,
                "support_identity": identities["support"], "test_identity": identities["test"],
                "split_sha256": {str(splits / "Toy" / f"{split}.csv"):
                                 sha256(splits / "Toy" / f"{split}.csv")
                                 for split in ("support", "test")},
                "auxiliary_files": {path.name: {"sha256": sha256(path)}
                                    for path in (cache / "support_labels.npy",
                                                 cache / "support_metadata.json",
                                                 cache / "test_metadata.json")},
            }
            (cache / "manifest.json").write_text(json.dumps(manifest))
            chosen_configs, predictions = [], []
            for flip_test_labels in (False, True):
                output = root / ("flipped" if flip_test_labels else "first")
                args = parse_args(["--cache", str(cache), "--output", str(output),
                                   "--dataset", "Toy", "--methods", "proto",
                                   "--features", "class_local"])
                test_reads = []

                def load(directory, name, split):
                    if split == "test":
                        self.assertTrue((output / "selection.json").exists())
                        test_reads.append(split)
                        result = dict(specs[split])
                        if flip_test_labels:
                            result["labels"] = 1 - result["labels"]
                        return result
                    return specs[split]

                with patch("i2cl_data.load_spec", side_effect=load):
                    run(args)
                self.assertEqual(test_reads, ["test"])
                selection = json.loads((output / "selection.json").read_text())
                chosen = selection["candidates"][0]
                self.assertEqual(chosen["selection_score"], 1.)
                self.assertGreater(chosen["config"]["strength"], 0.)
                self.assertLess(chosen["support_oof_bce"]["weighted_bce"],
                                selection["baseline_support_oof_bce"]["weighted_bce"])
                chosen_configs.append(chosen["config"])
                predictions.append(np.load(output / chosen["candidate_id"] / "predictions.npy"))
                self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
            self.assertEqual(chosen_configs[0], chosen_configs[1])
            np.testing.assert_array_equal(predictions[0], predictions[1])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
