"""CPU-only checks for the declared support-query candidate family."""
import ast
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from query_candidates import CANDIDATE_NAMES, fit_candidates


def synthetic(n=10, c=3, d=24):
    generator = torch.Generator().manual_seed(123)
    features = F.normalize(torch.randn(n, c, d, generator=generator), dim=-1)
    labels = torch.tensor([[(i + j) % 2 for j in range(c)] for i in range(n)], dtype=torch.float32)
    original = torch.randn(c, d, generator=generator)
    return features, labels, original


class CandidateTests(unittest.TestCase):
    def test_exact_existing_radzero_formula_and_raw_zero(self):
        root = Path(__file__).resolve().parents[3]
        source = root / "RadZero_i2cl/local_setup/overnight_query_refinement.py"
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"eta_name", "validate_etas", "fit_query_directions"}]
        namespace = {"np": np, "torch": torch, "F": F, "math": math}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                     str(source), "exec"), namespace)
        x, y, q = synthetic()
        expected = namespace["fit_query_directions"](x.numpy(), y.numpy(), q.numpy(), (0., .3))
        actual = fit_candidates(x, y, q)
        np.testing.assert_array_equal(actual["queries"][:2], expected["queries"])
        np.testing.assert_array_equal(actual["queries"][0], q.numpy())
        self.assertEqual(actual["names"], list(CANDIDATE_NAMES))
        self.assertEqual(actual["queries"].dtype, np.float32)
        json.dumps(actual["metadata"], allow_nan=False)

    def test_multihead_geometry_and_no_input_mutation(self):
        x, y, q = synthetic(c=3, d=768)
        before = [value.clone() for value in (x, y, q)]
        result = fit_candidates(x, y, q, num_heads=8)
        self.assertEqual(result["queries"].shape, (10, 3, 768))
        for actual, expected in zip((x, y, q), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        original = q.reshape(3, 8, 96)
        for name in ("head_tangent_eta03", "head_tangent_variance_eta03_k1"):
            new = torch.from_numpy(result["queries"][result["names"].index(name)]).reshape_as(original)
            torch.testing.assert_close(new.norm(dim=-1), original.norm(dim=-1), rtol=2e-6, atol=2e-6)
            cosine = (F.normalize(new, dim=-1) * F.normalize(original, dim=-1)).sum(-1).clamp(-1, 1)
            self.assertTrue(bool((torch.acos(cosine) <= math.atan(.3) + 2e-5).all()))

    def test_single_class_ridge_falls_back_exactly(self):
        x, y, q = synthetic(c=1)
        result = fit_candidates(x, y, q)
        np.testing.assert_array_equal(result["queries"][1], result["queries"][-1])
        self.assertEqual(result["metadata"][-1]["diagnostics"][0]["fallback"], "single-class task")

    def test_missing_classes_and_invalid_inputs_rejected(self):
        x, y, q = synthetic()
        for label in (torch.zeros_like(y), torch.ones_like(y), y * float("nan")):
            with self.assertRaises(ValueError):
                fit_candidates(x, label, q)
        with self.assertRaises(ValueError):
            fit_candidates(x, y, q, num_heads=7)
        with self.assertRaises(ValueError):
            fit_candidates(x[:0], y[:0], q)
        with self.assertRaises(ValueError):
            fit_candidates(x * float("nan"), y, q)
        with self.assertRaises(ValueError):
            fit_candidates(x, y, q, groups=[None] * len(x))

    def test_whole_group_leaveout_abstains_when_unidentifiable(self):
        x, y, q = synthetic(c=1)
        groups = ["positive_patient" if label else "negative_patient" for label in y[:, 0]]
        result = fit_candidates(x, y, q, num_heads=3, groups=groups)
        for name in ("global_variance_eta03_k1", "global_loo_eta03", "head_tangent_variance_eta03_k1"):
            index = result["names"].index(name)
            np.testing.assert_array_equal(result["queries"][index], q.numpy())
        self.assertEqual(result["metadata"][6]["folds"], 2)
        self.assertEqual(result["metadata"][6]["valid_fold_counts"], [0])

    def test_constant_features_and_zero_heads_are_finite(self):
        x, y, q = synthetic()
        x[:] = x[0]
        q[0, :8] = 0
        result = fit_candidates(x, y, q, num_heads=3)
        self.assertTrue(np.isfinite(result["queries"]).all())
        for index in (7, 8):
            np.testing.assert_array_equal(result["queries"][index, 0, :8], np.zeros(8, dtype=np.float32))

    def test_group_permutation_and_perturbed_features(self):
        x, y, q = synthetic(n=12)
        groups = np.asarray([i // 2 for i in range(12)])
        original = fit_candidates(x, y, q, num_heads=3, groups=groups)
        permutation = np.random.default_rng(42).permutation(len(x))
        changed = fit_candidates(x[permutation], y[permutation], q, num_heads=3, groups=groups[permutation])
        np.testing.assert_allclose(changed["queries"], original["queries"], atol=3e-6, rtol=3e-6)
        generator = torch.Generator().manual_seed(42)
        disturbed = F.normalize(x + .01 * torch.randn(x.shape, generator=generator), dim=-1)
        result = fit_candidates(disturbed, y, q, num_heads=3, groups=groups)
        self.assertTrue(np.isfinite(result["queries"]).all())
        json.dumps(result["metadata"], allow_nan=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
