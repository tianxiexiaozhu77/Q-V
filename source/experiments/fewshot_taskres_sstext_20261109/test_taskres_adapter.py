"""CPU mathematical, protocol, and serialization checks; no dataset evaluation."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .taskres_adapter import TaskResModel, _fit_path, fit_taskres, learning_rate_at_epoch
except ImportError:
    from taskres_adapter import TaskResModel, _fit_path, fit_taskres, learning_rate_at_epoch


def unit(value):
    value = np.asarray(value, dtype=np.float32)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def upstream_learner():
    """Load the exact pure author class without importing Dassl/CLIP or CUDA."""
    root = Path(__file__).resolve().parents[2]
    source = root / "baselines/TaskRes/trainers/taskres.py"
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskResLearner"]
    if len(definitions) != 1:
        raise AssertionError("expected one upstream TaskResLearner")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["TaskResLearner"]


class TaskResTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny CPU tensor operations are needlessly slow with server-wide BLAS
        # threading. Restore the setting so importing/running tests is isolated.
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        rng = np.random.default_rng(120)
        self.x = unit(rng.normal(size=(20, 8)))
        # Deliberately unequal nonunit base norms: premature normalization
        # changes TaskRes's residual strength and must fail parity tests.
        self.qpos = unit(rng.normal(size=(2, 8))) * np.array([[4.6], [3.7]], dtype=np.float32)
        self.qneg = unit(rng.normal(size=(2, 8))) * np.array([[5.3], [2.8]], dtype=np.float32)
        self.y = np.stack((np.arange(20) % 2, (np.arange(20) // 2) % 2), axis=1)
        self.groups = [f"p-{i}" for i in range(20)]
        self.folds = [{"train": np.flatnonzero(np.arange(20) % 5 != fold),
                       "validation": np.flatnonzero(np.arange(20) % 5 == fold)} for fold in range(5)]
        self.fit_options = dict(logit_scale=7.2, epoch_candidates=(0, 2, 4),
                                learning_rate_candidates=(0.002,), schedule_epochs=4, seed=94)

    def test_upstream_residual_forward_and_gradient_parity(self):
        model = TaskResModel(self.qpos, self.qneg, logit_scale=7.2, residual_scale=0.5)
        base = torch.from_numpy(np.stack((self.qneg, self.qpos), axis=1)).reshape(-1, 8)
        config = SimpleNamespace(TRAINER=SimpleNamespace(TaskRes=SimpleNamespace(RESIDUAL_SCALE=0.5)))
        author = upstream_learner()(config, [], SimpleNamespace(dtype=torch.float32), base)
        perturbation = torch.from_numpy(np.random.default_rng(81).normal(size=(2, 2, 8)).astype(np.float32)) * 0.04
        with torch.no_grad():
            model.text_feature_residuals.copy_(perturbation)
            author.text_feature_residuals.copy_(perturbation.reshape(-1, 8))
        x = torch.from_numpy(self.x)
        actual = model(x)
        author_text = author()
        author_text = author_text / author_text.norm(dim=-1, keepdim=True)
        author_x = x / x.norm(dim=-1, keepdim=True)
        expected = (7.2 * author_x @ author_text.t()).reshape(20, 2, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        y = torch.from_numpy(self.y)
        own_loss = F.cross_entropy(actual.reshape(-1, 2), y.reshape(-1), reduction="none").reshape(20, 2).mean(0).sum()
        author_loss = sum(F.cross_entropy(expected[:, c], y[:, c]) for c in range(2))
        own_loss.backward()
        author_loss.backward()
        torch.testing.assert_close(model.text_feature_residuals.grad.reshape(-1, 8), author.text_feature_residuals.grad,
                                   rtol=2e-6, atol=1e-6)
        self.assertIsNone(model.base_text_features.grad)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2 * 2 * 8)

    def test_one_step_matches_author_with_official_adam(self):
        model, _, _ = _fit_path(self.x, self.y, self.qpos, self.qneg,
                                checkpoints=(1,), learning_rate=0.002,
                                device="cpu", seed=71, residual_scale=0.5, logit_scale=7.2,
                                schedule_epochs=200, batch_size=256, weight_decay=0.0005, warmup_lr=1e-5)
        base = torch.from_numpy(np.stack((self.qneg, self.qpos), axis=1)).reshape(-1, 8)
        config = SimpleNamespace(TRAINER=SimpleNamespace(TaskRes=SimpleNamespace(RESIDUAL_SCALE=0.5)))
        author = upstream_learner()(config, [], SimpleNamespace(dtype=torch.float32), base)
        optimizer = torch.optim.Adam(author.parameters(), lr=1e-5, weight_decay=0.0005, betas=(0.9, 0.999), eps=1e-8)
        text = F.normalize(author(), dim=-1)
        logits = (7.2 * F.normalize(torch.from_numpy(self.x), dim=-1) @ text.t()).reshape(20, 2, 2)
        loss = sum(F.cross_entropy(logits[:, c], torch.from_numpy(self.y[:, c])) for c in range(2))
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(model.text_feature_residuals.reshape(-1, 8), author.text_feature_residuals,
                                   atol=1e-10, rtol=2e-6)

    def test_independent_ovr_matches_separately_fitted_binary_tasks(self):
        opts = dict(checkpoints=(4,), learning_rate=0.002, device="cpu", seed=89,
                    residual_scale=0.5, logit_scale=7.2, schedule_epochs=200,
                    batch_size=256, weight_decay=0.0005, warmup_lr=1e-5)
        joint, _, _ = _fit_path(self.x, self.y, self.qpos, self.qneg, **opts)
        for c in range(2):
            separate, _, _ = _fit_path(self.x, self.y[:, c:c+1], self.qpos[c:c+1], self.qneg[c:c+1], **opts)
            torch.testing.assert_close(joint.text_feature_residuals[c], separate.text_feature_residuals[0], rtol=2e-6, atol=1e-8)

    def test_heldout_arrays_cannot_change_fold_optimization(self):
        train = self.folds[0]["train"]
        held = self.folds[0]["validation"]
        altered_x, altered_y = self.x.copy(), self.y.copy()
        altered_x[held] *= -1
        altered_y[held] = 1 - altered_y[held]
        opts = dict(checkpoints=(2,), learning_rate=0.002, device="cpu", seed=33,
                    residual_scale=0.5, logit_scale=7.2, schedule_epochs=200,
                    batch_size=256, weight_decay=0.0005, warmup_lr=1e-5)
        model_a, _, _ = _fit_path(self.x[train], self.y[train], self.qpos, self.qneg, **opts)
        model_b, _, _ = _fit_path(altered_x[train], altered_y[train], self.qpos, self.qneg, **opts)
        torch.testing.assert_close(model_a.text_feature_residuals, model_b.text_feature_residuals, rtol=0, atol=0)
        self.assertNotIn("test_labels", inspect.signature(fit_taskres).parameters)
        self.assertNotIn("test_features", inspect.signature(fit_taskres).parameters)
        with self.assertRaises(TypeError):
            fit_taskres(self.x, self.y, self.qpos, self.qneg, self.groups, self.folds, test_labels=self.y, **self.fit_options)

    def test_deterministic_support_selection_and_export(self):
        x_before, y_before = self.x.copy(), self.y.copy()
        torch_state_before = torch.random.get_rng_state().clone()
        model_a, info_a = fit_taskres(self.x, self.y, self.qpos, self.qneg, self.groups, self.folds, **self.fit_options)
        model_b, info_b = fit_taskres(self.x, self.y, self.qpos, self.qneg, self.groups, self.folds, **self.fit_options)
        self.assertEqual(info_a["candidates"], info_b["candidates"])
        self.assertEqual(info_a["selected"], info_b["selected"])
        np.testing.assert_array_equal(model_a.predict(self.x), model_b.predict(self.x))
        restored = TaskResModel.from_export(json.loads(json.dumps(model_a.export())))
        np.testing.assert_array_equal(model_a.predict(self.x), restored.predict(self.x))
        np.testing.assert_allclose(model_a.predict(self.x, batch_size=3), model_a.predict(self.x), rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(self.x, x_before)
        np.testing.assert_array_equal(self.y, y_before)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state_before, rtol=0, atol=0)
        json.dumps(info_a, allow_nan=False)

    def test_checkpoint_does_not_retime_cosine_schedule(self):
        opts = dict(learning_rate=0.002, device="cpu", seed=61,
                    residual_scale=0.5, logit_scale=7.2, schedule_epochs=200,
                    batch_size=256, weight_decay=0.0005, warmup_lr=1e-5)
        _, snapshot_short, trace_short = _fit_path(self.x, self.y, self.qpos, self.qneg, checkpoints=(2,), **opts)
        _, snapshot_long, trace_long = _fit_path(self.x, self.y, self.qpos, self.qneg, checkpoints=(2, 4), **opts)
        torch.testing.assert_close(snapshot_short[2], snapshot_long[2], rtol=0, atol=0)
        self.assertEqual(trace_short["epoch_trace"], trace_long["epoch_trace"][:2])
        self.assertEqual(learning_rate_at_epoch(0, 0.002), 1e-5)
        self.assertEqual(learning_rate_at_epoch(1, 0.002), 0.002)
        self.assertAlmostEqual(learning_rate_at_epoch(101, 0.002), 0.001)

    def test_leaking_groups_invalid_features_and_missing_sides_rejected(self):
        leaking = self.groups.copy()
        leaking[0] = leaking[1]
        with self.assertRaisesRegex(ValueError, "crosses"):
            fit_taskres(self.x, self.y, self.qpos, self.qneg, leaking, self.folds, **self.fit_options)
        with self.assertRaisesRegex(ValueError, "unit vectors"):
            fit_taskres(self.x * 2, self.y, self.qpos, self.qneg, self.groups, self.folds, **self.fit_options)
        rare = self.y.copy()
        rare[:, 0] = 0
        rare[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "lack one binary label"):
            fit_taskres(self.x, rare, self.qpos, self.qneg, self.groups, self.folds, **self.fit_options)

    def test_zero_epoch_is_native_scaled_cosine_margin(self):
        model, info = fit_taskres(self.x, self.y, self.qpos, self.qneg, self.groups, self.folds,
                                 logit_scale=7.2, epoch_candidates=(0,))
        expected = 7.2 * (self.x @ unit(self.qpos).T - self.x @ unit(self.qneg).T)
        np.testing.assert_allclose(model.predict(self.x), expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(info["selected"]["epoch"], 0)
        self.assertEqual(info["optimization_paths"], [])


if __name__ == "__main__":
    unittest.main()
