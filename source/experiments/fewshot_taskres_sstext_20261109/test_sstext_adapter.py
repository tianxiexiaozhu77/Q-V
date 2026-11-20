"""CPU parity tests against the actual checked-out official SS-Text modules."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
from pathlib import Path
import unittest
import warnings

import numpy as np
import torch

from sstext_adapter import SOURCE_COMMIT, SOURCE_SHA256, SStextModel, fit_sstext


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "baselines" / "SS-Text"


def source_module(name, relative_path):
    """Load executable upstream code directly; do not replace its algorithm."""
    path = UPSTREAM / relative_path
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256[relative_path]:
        raise AssertionError(f"Upstream source differs from pinned commit: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Execute the real source without writing bytecode into the upstream clone.
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


class TestSStextParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.is_available():
            raise RuntimeError("Run parity tests with CUDA_VISIBLE_DEVICES='' to ensure CPU-only execution")
        cls.solver_module = source_module("official_sstext", "solvers/sstext.py")
        cls.adapter_module = source_module("official_sstext_models", "modeling/adapters/models.py")
        # These small fixtures should not create excessive CPU thread pools.
        torch.set_num_threads(1)

    def fixture(self, counts, d=13, seed=841):
        rng = np.random.default_rng(seed)
        n = len(counts[0])
        labels = np.array(counts, dtype=np.int64).T
        # Deliberately unequal norms detect extra normalization in the port.
        features = rng.normal(size=(n, d)).astype(np.float32) * rng.uniform(0.2, 3.0, (n, 1))
        positive = rng.normal(size=(len(counts), d)).astype(np.float32) * 0.7
        negative = rng.normal(size=(len(counts), d)).astype(np.float32) * 1.9
        test_features = rng.normal(size=(9, d)).astype(np.float32) * 2.3
        return features.astype(np.float32), labels, positive, negative, test_features

    def assert_official_parity(self, data, *, logit_scale=2.2, balanced_importance=False, repel=True):
        features, labels, positive, negative, test_features = data
        model, info = fit_sstext(
            features, labels, positive, negative, logit_scale=logit_scale,
            balanced_importance=balanced_importance, repel=repel,
        )
        expected_margins, expected_prototypes = [], []
        for disease in range(labels.shape[1]):
            initial = torch.tensor(np.stack([negative[disease], positive[disease]]))
            official = self.adapter_module.Adapter(initial, float(logit_scale), adapter="SStext+")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # reciprocal count for missing classes
                _, official = self.solver_module.ss_text_solver(
                    torch.tensor(features), torch.tensor(labels[:, disease]), official,
                    classes=np.array([0, 1]), text_lambda="adaptive_shots",
                    balanced_importance=balanced_importance, repel=repel,
                )
            with torch.no_grad():
                logits = official(torch.tensor(test_features))
            expected_margins.append((logits[:, 1] - logits[:, 0]).numpy())
            expected_prototypes.append(official.adapter.prototypes.detach().numpy())
        np.testing.assert_array_equal(model.export()["prototypes"], np.stack(expected_prototypes))
        np.testing.assert_array_equal(model.predict(test_features), np.stack(expected_margins, axis=1))
        json.dumps(info, allow_nan=False)
        self.assertEqual(info["source_commit"], SOURCE_COMMIT)
        self.assertFalse(info["validation_used"])
        self.assertFalse(info["test_information_used"])
        return model, info

    def test_balanced_binary_exact_upstream_parity(self):
        self.assert_official_parity(self.fixture([[0, 1] * 5]))

    def test_imbalanced_overlapping_multilabel_exact_upstream_parity(self):
        data = self.fixture([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ])
        _, info = self.assert_official_parity(data)
        self.assertEqual(info["support_counts"], [[10, 1], [8, 3], [1, 10]])
        self.assertEqual(info["method"], "SStext+")

    def test_missing_support_class_matches_official_realistic_scenario(self):
        data = self.fixture([[0] * 9, [1] * 9])
        _, info = self.assert_official_parity(data)
        self.assertIsNone(info["lambda_text"][0][1])
        self.assertIsNone(info["lambda_text"][1][0])

    def test_official_lambda_floor_is_not_omitted(self):
        data = self.fixture([[0] * 1003 + [1] * 2], d=5)
        _, info = self.assert_official_parity(data)
        self.assertAlmostEqual(info["lambda_text"][0][0], 1e-3, places=9)

    def test_official_solver_options_preserve_operation_order(self):
        data = self.fixture([[0] * 7 + [1] * 3])
        for balanced, repel in [(True, False), (True, True), (False, False)]:
            with self.subTest(balanced=balanced, repel=repel):
                self.assert_official_parity(data, balanced_importance=balanced, repel=repel)

    def test_npz_roundtrip_and_independent_test_batches(self):
        data = self.fixture([[0, 1] * 5, [1] * 3 + [0] * 7])
        model, _ = self.assert_official_parity(data)
        expected = model.predict(data[-1])
        before = model.export()["prototypes"].copy()
        buffer = io.BytesIO()
        np.savez(buffer, **model.export())
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as state:
            restored = SStextModel.from_export(state)
        np.testing.assert_array_equal(restored.predict(data[-1]), expected)
        by_row = np.concatenate([model.predict(row[None]) for row in data[-1]], axis=0)
        np.testing.assert_allclose(by_row, expected, rtol=2e-6, atol=2e-6)
        model.predict(data[-1] * -7)
        np.testing.assert_array_equal(model.export()["prototypes"], before)
        self.assertEqual(model.predict(np.empty((0, data[0].shape[1]), dtype=np.float32)).shape, (0, 2))

    def test_no_test_information_in_fit_api_and_no_input_mutation(self):
        data = self.fixture([[0, 1] * 4])
        originals = [x.copy() for x in data[:4]]
        signature = inspect.signature(fit_sstext)
        self.assertNotIn("test_features", signature.parameters)
        self.assertNotIn("test_labels", signature.parameters)
        self.assertNotIn("validation", signature.parameters)
        model, _ = fit_sstext(*data[:4], logit_scale=1.0)
        for original, current in zip(originals, data[:4]):
            np.testing.assert_array_equal(original, current)
        exported = model.export()
        exported["prototypes"][:] = 0
        self.assertTrue(np.any(model.export()["prototypes"] != 0))

    def test_invalid_inputs_fail_explicitly(self):
        data = self.fixture([[0, 1] * 4])
        features, labels, positive, negative, _ = data
        with self.assertRaisesRegex(ValueError, "binary"):
            fit_sstext(features, labels - 1, positive, negative, logit_scale=1.0)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            fit_sstext(features * 0, labels, positive, negative, logit_scale=1.0)
        with self.assertRaisesRegex(ValueError, "adaptive_shots"):
            fit_sstext(features, labels, positive, negative, logit_scale=1.0, text_lambda="adaptive_perf")
        with self.assertRaisesRegex(ValueError, "shape"):
            fit_sstext(features, labels[:, 0], positive, negative, logit_scale=1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_sstext(features, labels, positive, negative, logit_scale=np.inf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
