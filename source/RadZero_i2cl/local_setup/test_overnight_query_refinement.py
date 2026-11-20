#!/usr/bin/env python3
"""CPU-only tests for support-derived queries and original RadZero attention."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from overnight_query_refinement import (
    DEFAULT_ETAS, HARD_DEADLINE, DeadlineReached, QueryRefinementMatcher,
    check_deadline, fit_query_directions, fit_reaggregation_query_directions,
    fit_uncertainty_query_directions,
    parse_deadline, validate_etas,
    verified_cache,
)
from extract_frozen_features import fingerprint, sha256
from test_i2cl_model import TinyRadZero


class CountingRadZero(TinyRadZero):
    def __init__(self):
        super().__init__()
        self.vision_calls = 0
        self.loss_fns["KeyPhraseAlignmentLoss"].sim_op = "cos"
        self.loss_fns["KeyPhraseAlignmentLoss"].similarity_logit.sim_op = "cos"

    def forward_vision_model(self, pixels):
        self.vision_calls += 1
        return {"vision_tokens": self.align_transformer(
            self.vision_model(pixels)["last_hidden_state"]
        )}


class QueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_direction_is_positive_mean_minus_negative_mean(self):
        features = np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[0., -1.]]], np.float32)
        labels = np.array([[1], [1], [0], [0]], np.int8)
        query = np.array([[2., 1.]], np.float32)
        state = fit_query_directions(features, labels, query, DEFAULT_ETAS)
        np.testing.assert_allclose(state["directions"], [[2 ** -.5, 2 ** -.5]], rtol=1e-6)
        np.testing.assert_array_equal(state["queries"][0], query)
        np.testing.assert_array_equal(state["positive_counts"], [2])
        np.testing.assert_array_equal(state["negative_counts"], [2])
        np.testing.assert_allclose(np.linalg.norm(state["queries"][1:], axis=-1), 1., atol=1e-6)

    def test_zero_eta_exact_identity_and_single_visual_forward_for_grid(self):
        for batch in (1, 3):
            for classes in (1, 2):
                with self.subTest(batch=batch, classes=classes):
                    base = CountingRadZero().eval()
                    original = torch.randn(classes, 16)
                    directions = F.normalize(torch.randn(classes, 16), dim=-1)
                    queries = torch.stack([
                        original, F.normalize(F.normalize(original, dim=-1) + .3 * directions, dim=-1)
                    ])
                    matcher = QueryRefinementMatcher(base, queries, [0., .3])
                    pixels = torch.randn(batch, 5, 16)
                    outputs = matcher(pixels, return_maps=True)
                    self.assertEqual(base.vision_calls, 1)
                    self.assertEqual(outputs[0.]["logits"].shape, (batch, classes))
                    self.assertEqual(outputs[.3]["similarity_scores"].shape, (batch, classes, 2, 2))
                    self.assertFalse(torch.equal(
                        outputs[0.]["similarity_scores"], outputs[.3]["similarity_scores"]
                    ))
                    with torch.no_grad():
                        prepared = matcher.prepare_vision(base.align_transformer(pixels))
                        logits, attention = base.loss_fns["KeyPhraseAlignmentLoss"].compute_t2i_logits(
                            original, prepared, need_attn_weights=True
                        )
                        logits = logits.reshape(classes, batch).T / matcher.loss.loss_temperature.exp()
                    torch.testing.assert_close(outputs[0.]["logits"], logits, atol=0, rtol=0)
                    torch.testing.assert_close(
                        outputs[0.]["similarity_scores"],
                        attention[0][:, :, 1:].reshape(batch, classes, 2, 2), atol=0, rtol=0,
                    )
                    self.assertTrue(all(not p.requires_grad for p in base.parameters()))
                    self.assertTrue(all(not value.requires_grad for item in outputs.values()
                                        for value in item.values()))

    def test_readout_only_uses_original_attention_and_updated_readout(self):
        base = CountingRadZero().eval()
        original = torch.randn(2, 16)
        directions = F.normalize(torch.randn(2, 16), dim=-1)
        updated = F.normalize(F.normalize(original, dim=-1) + .3 * directions, dim=-1)
        queries = torch.stack([original, updated])
        matcher = QueryRefinementMatcher(base, queries, [0., .3], readout_only=True)
        pixels = torch.randn(3, 5, 16)
        outputs = matcher(pixels)
        prepared = matcher.prepare_vision(base.align_transformer(pixels))
        readout = matcher.match_readout_only_prepared(prepared, original, updated)["logits"]
        torch.testing.assert_close(outputs[.3]["logits"], readout, atol=0, rtol=0)
        # The q0 path remains the exact native scorer and the adapted path is
        # generally different from One-step when token attention changes.
        native = matcher.match_prepared(prepared, original)["logits"]
        self.assertEqual(outputs[0.]["logits"].shape, (3, 2))
        torch.testing.assert_close(outputs[0.]["logits"], native, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(outputs[.3]["logits"]).all())

    def test_missing_class_and_nonfinite_support_rejected(self):
        features = np.ones((4, 1, 3), np.float32)
        query = np.ones((1, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            fit_query_directions(features, np.ones((4, 1)), query, DEFAULT_ETAS)
        features[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fit_query_directions(features, [[0], [0], [1], [1]], query, DEFAULT_ETAS)

    def test_zero_support_direction_stays_finite_and_recorded(self):
        state = fit_query_directions(
            np.ones((4, 1, 3), np.float32), [[0], [0], [1], [1]],
            [[1., 2., 3.]], DEFAULT_ETAS,
        )
        self.assertTrue(state["zero_direction_classes"][0])
        np.testing.assert_array_equal(state["directions"], np.zeros((1, 3)))
        self.assertTrue(np.isfinite(state["queries"]).all())

    def test_uncertainty_update_is_finite_and_reliability_bounded(self):
        features = F.normalize(torch.randn(8, 2, 12), dim=-1).numpy()
        labels = np.array([[1, 0], [1, 0], [1, 0], [1, 0],
                           [0, 1], [0, 1], [0, 1], [0, 1]], np.int8)
        query = F.normalize(torch.randn(2, 12), dim=-1).numpy()
        state = fit_uncertainty_query_directions(features, labels, query, DEFAULT_ETAS)
        np.testing.assert_array_equal(state["queries"][0], query)
        np.testing.assert_allclose(np.linalg.norm(state["queries"][1:], axis=-1), 1., atol=1e-6)
        self.assertTrue(np.isfinite(state["queries"]).all())
        self.assertTrue(np.all((state["support_reliability"] >= 0) &
                               (state["support_reliability"] <= 1)))

    def test_reaggregation_recomputes_support_attention_and_stops(self):
        base = CountingRadZero().eval()
        original = torch.randn(2, 16)
        matcher = QueryRefinementMatcher(base, original[None], [0.])
        aligned = torch.randn(6, 5, 16)
        labels = np.array([[1, 0], [1, 0], [1, 0],
                           [0, 1], [0, 1], [0, 1]], np.int8)
        state = fit_reaggregation_query_directions(
            [aligned[:3], aligned[3:]], labels, original.numpy(), matcher,
            eta=.3, max_iter=2, query_tol=1e-6)
        self.assertEqual(state["queries"].shape, (2, 2, 16))
        self.assertEqual(state["reaggregation_iterations"], 2)
        self.assertEqual(state["reaggregation_stop_reason"], "max_iter")
        self.assertEqual(state["query_trajectory"].shape, (3, 2, 16))
        self.assertTrue(all("support_utility" in item and
                            "local_lipschitz_max" in item
                            for item in state["reaggregation_history"]))
        self.assertTrue(np.isfinite(state["queries"]).all())
        np.testing.assert_allclose(np.linalg.norm(state["queries"][1], axis=-1), 1., atol=1e-6)

    def test_grid_requires_identity_and_no_duplicates(self):
        self.assertEqual(validate_etas(DEFAULT_ETAS), DEFAULT_ETAS)
        for invalid in ([.1], [0, 0], [0, np.nan], [0, -.1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_etas(invalid)

    def test_deadline_is_timezone_aware_and_cannot_extend_past_authorization(self):
        self.assertEqual(parse_deadline("2026-10-21T10:00:00+08:00"), HARD_DEADLINE)
        self.assertEqual(parse_deadline(str(HARD_DEADLINE - 30)), HARD_DEADLINE - 30)
        with self.assertRaisesRegex(ValueError, "timezone"):
            parse_deadline("2026-10-21T10:00:00")
        with self.assertRaises(ValueError):
            parse_deadline("2026-10-21T10:01:00+08:00")
        with self.assertRaises(DeadlineReached):
            check_deadline(0.)

    def test_verified_cache_consumes_exact_schema_and_rejects_mutated_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "preprocessor_config.json").write_text("{}")
            support = {"image_paths": ["a", "b"], "groups": ["p1", "p2"],
                       "ids": ["a", "b"], "class_names": ["x"], "prompts": ["There is x"],
                       "name": "test", "data_root": "/data"}
            test = dict(support, image_paths=["c"], groups=["p3"], ids=["c"])
            manifest = {
                "status": "complete", "auxiliary_files": {}, "splits": {},
                "split_sha256": {},
                "model_config_sha256": sha256(model / "config.json"),
                "preprocessor_sha256": sha256(model / "preprocessor_config.json"),
            }
            for split, metadata in (("support", support), ("test", test)):
                path = root / f"{split}_metadata.json"
                path.write_text(json.dumps(metadata))
                manifest["auxiliary_files"][path.name] = {"sha256": sha256(path)}
                manifest[f"{split}_identity"] = fingerprint(metadata)
            for split, name, values in (
                ("support", "class_local", np.ones((2, 1, 3), np.float32)),
                ("test", "base_logits", np.ones((1, 1), np.float32)),
            ):
                path = root / f"{split}_{name}.npy"
                np.save(path, values)
                manifest["splits"][split] = {"files": {name: {
                    "file": path.name, "sha256": sha256(path),
                    "shape": list(values.shape), "dtype": str(values.dtype),
                }}}
            for name, values in (("support_labels", np.array([[0], [1]], np.int8)),
                                 ("text_features", np.ones((1, 3), np.float32))):
                path = root / f"{name}.npy"
                np.save(path, values)
                manifest["auxiliary_files"][path.name] = {"sha256": sha256(path)}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch("overnight_query_refinement.MODEL_DIR", model):
                result = verified_cache(root)
                np.testing.assert_array_equal(result["support_labels"], [[0], [1]])
                self.assertNotIn("labels", result["test_metadata"])
                np.save(root / "support_class_local.npy", np.zeros((2, 1, 3), np.float32))
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verified_cache(root)


if __name__ == "__main__":
    unittest.main()
