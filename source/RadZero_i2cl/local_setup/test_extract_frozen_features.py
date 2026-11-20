#!/usr/bin/env python3
"""CPU-only checks for compact frozen feature caching; no pretrained loading."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from extract_frozen_features import (
    FrozenFeatureExtractor, assert_disjoint, cache_split, feature_shapes, pool_original_matching,
    public_spec, read_baseline, strict_baseline_check,
)
from test_i2cl_model import TinyRadZero, tokenizer


class FeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_pooled_vectors_reconstruct_official_logits_including_singletons(self):
        for classes in (1, 3):
            for include_cls in (False, True):
                for batch in (1, 4):
                    with self.subTest(classes=classes, cls=include_cls, batch=batch):
                        base = TinyRadZero().eval()
                        loss = base.loss_fns["KeyPhraseAlignmentLoss"]
                        loss.sim_op = "cos"
                        loss.similarity_logit.sim_op = "cos"
                        loss.use_vision_cls_token = include_cls
                        wrapper = FrozenFeatureExtractor(
                            base, tokenizer, [f"finding {i}" for i in range(classes)]
                        )
                        tokens = torch.randn(batch, 5, 16)
                        output = wrapper(tokens)
                        self.assertEqual(output["class_local"].shape, (batch, classes, 16))
                        self.assertEqual(output["global_dino"].shape, (batch, 2, 16))
                        self.assertEqual(output["base_logits"].shape, (batch, classes))
                        torch.testing.assert_close(
                            output["class_local"].norm(dim=-1),
                            torch.ones(batch, classes), atol=2e-6, rtol=0,
                        )
                        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
                        self.assertTrue(all(not x.requires_grad for x in output.values()))
                        torch.testing.assert_close(output["global_dino"][:, 0], tokens[:, 0])
                        torch.testing.assert_close(
                            output["global_dino"][:, 1], tokens[:, 1:].mean(1)
                        )

    def test_rejects_changed_scoring_semantics(self):
        base = TinyRadZero()
        loss = base.loss_fns["KeyPhraseAlignmentLoss"]
        loss.sim_op = "dot"
        with self.assertRaisesRegex(ValueError, "cosine"):
            pool_original_matching(loss, torch.randn(2, 16), torch.randn(3, 5, 16))

    def test_test_metadata_does_not_export_labels_or_positive_counts(self):
        spec = {
            "name": "test", "data_root": "/data", "image_paths": ["a"], "ids": ["a"],
            "groups": ["p1"], "class_names": ["x"], "prompts": ["There is x"],
            "labels": np.ones((1, 1)), "metadata": {"positive_images_per_class": {"x": 1}},
        }
        safe = public_spec(spec)
        self.assertNotIn("labels", safe)
        self.assertNotIn("metadata", safe)
        other = dict(safe, image_paths=["b"], ids=["b"], groups=["p2"])
        assert_disjoint(safe, other)
        with self.assertRaisesRegex(ValueError, "groups"):
            assert_disjoint(safe, dict(other, groups=["p1"]))

    def test_historical_csv_labels_not_parsed_and_order_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["index", "image_path", "label:x", "logit:x"])
                writer.writerow([0, "a", "must_not_parse_this_label", 0.123])
                writer.writerow([1, "b", "must_not_parse_this_label", -0.2])
            values = read_baseline(path, ["a", "b"], ["x"])
            np.testing.assert_array_equal(values, np.array([[.123], [-.2]], np.float32))
            with self.assertRaisesRegex(ValueError, "order"):
                read_baseline(path, ["b", "a"], ["x"])

    def test_baseline_check_absolute_tolerance_and_last_batch_shapes(self):
        original = np.array([[1., -2.], [.5, -.25]], np.float32)
        self.assertTrue(strict_baseline_check(original, original)["bitwise_equal"])
        self.assertFalse(strict_baseline_check(original + 5e-7, original)["bitwise_equal"])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            strict_baseline_check(original + 1e-4, original)
        with self.assertRaisesRegex(ValueError, "shape"):
            strict_baseline_check(original[:1], original)
        shapes = feature_shapes(17, 3)
        self.assertEqual(shapes["class_local"], (17, 3, 768))
        self.assertEqual(shapes["base_logits"], (17, 3))

    def test_memmap_completeness_and_nonfull_last_batch(self):
        class FakeExtractor:
            def __call__(self, pixels):
                return {
                    "global_dino": torch.zeros(len(pixels), 2, 768),
                    "global_aligned": torch.ones(len(pixels), 2, 768),
                    "class_local": torch.ones(len(pixels), 1, 768) / (768 ** .5),
                    "base_logits": pixels.reshape(-1, 1).float(),
                }

            def verify_official_api(self, pixels):
                return {"passed": True}

        batches = [
            (torch.arange(start, stop), None, torch.arange(start, stop))
            for start, stop in ((0, 2), (2, 4), (4, 5))
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), device="cpu", dataset="SIIM")
            spec = {"image_paths": [str(i) for i in range(5)], "class_names": ["x"]}
            expected = np.arange(5, dtype=np.float32).reshape(-1, 1)
            with patch("extract_frozen_features.image_loader", return_value=batches):
                report = cache_split("test", spec, FakeExtractor(), None, args, expected)
            actual = np.load(args.output / "test_base_logits.npy", mmap_mode="r")
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(report["historical_baseline_check"]["checked_images"], 5)
            self.assertTrue(report["historical_baseline_check"]["bitwise_equal"])
            self.assertEqual(report["files"]["class_local"]["shape"], [5, 1, 768])
            progress = json.loads((args.output / "progress.json").read_text())
            self.assertEqual(progress["completed_images"], 5)


if __name__ == "__main__":
    unittest.main()
