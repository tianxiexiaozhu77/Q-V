"""CPU numerical/protocol checks for the explicitly named SimNL OvR transfer."""
import inspect
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from simnl_adapter import (
    BinarySimNL, SimNLConfig, _train_snapshots, adaptive_reranking_loss,
    confidence_cache_values, fit_simnl_ovr, negative_cache_values, validate_folds,
)


def fixture():
    rng = np.random.default_rng(421)
    features = rng.normal(size=(20, 7)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.stack((np.arange(20) % 2, (np.arange(20) // 2) % 2), axis=1).astype(np.float32)
    groups = np.arange(20) // 2
    queries = rng.normal(size=(2, 7)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    negative = rng.normal(size=(2, 7)).astype(np.float32)
    negative /= np.linalg.norm(negative, axis=1, keepdims=True)
    folds = [{"train": np.where(groups % 5 != k)[0].tolist(),
              "validation": np.where(groups % 5 == k)[0].tolist()} for k in range(5)]
    return features, labels, np.zeros_like(labels), queries, negative, groups, folds


class SimNLNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_confidence_weight_matches_source_for_equal_counts(self):
        x = F.normalize(torch.tensor([[1., .2, .4], [.3, .8, .2], [.4, .5, .7],
                                      [.1, -.5, .8], [.7, -.4, .1], [.9, .2, -.4]]), dim=1)
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        expected = F.one_hot(y, 2).float()
        for c in range(2):
            idx = y == c
            sim = x[idx] @ x[idx].T
            confidence = torch.softmax((sim.sum(1) - 1) / 2, dim=0)
            expected[idx] *= confidence[:, None] * 3
        torch.testing.assert_close(confidence_cache_values(x, y), expected)

    def test_unbalanced_cache_keeps_every_example_and_equal_total(self):
        x = F.normalize(torch.arange(1., 16.).reshape(5, 3), dim=1)
        y = torch.tensor([0, 0, 0, 0, 1])
        positive = confidence_cache_values(x, y)
        negative = negative_cache_values(y, x.dtype)
        self.assertEqual(positive.shape, (5, 2))
        torch.testing.assert_close(positive.sum(0), torch.tensor([2.5, 2.5]))
        torch.testing.assert_close(negative.sum(0), torch.tensor([2.5, 2.5]))
        self.assertTrue((positive.sum(1) > 0).all())
        self.assertTrue((negative.sum(1) > 0).all())
        self.assertTrue(torch.equal(negative.argmax(1), 1 - y))

    def test_balanced_complement_cache_matches_other_class_affinity(self):
        x, y, _, positive, negative, _, _ = fixture()
        x = torch.tensor(x[:6]); targets = torch.tensor([0, 0, 0, 1, 1, 1])
        q = x[:2].clone()
        # The official binary pseudo-negative keys exchange the two blocks.
        swapped = torch.cat((x[3:], x[:3]))
        source_hard_values = F.one_hot(targets, 2).float()
        ref = torch.exp(-(q @ swapped.T)) @ source_hard_values
        actual = torch.exp(-(q @ x.T)) @ negative_cache_values(targets, x.dtype)
        torch.testing.assert_close(actual, ref)

    def test_reranking_matches_direct_official_expression(self):
        x, y, _, positive, _, _, _ = fixture()
        x, q, y = torch.tensor(x), torch.tensor(positive), torch.tensor(y[:, 0]).long()
        distances = torch.cdist(x, q)
        ds, idx = distances.sort(dim=1)
        anchor = ((x - q[y]) ** 2).sum(-1).sqrt()[:, None]
        margins = (1 - (q[idx] * q[y, None]).sum(-1)) / 4
        ref = torch.maximum(anchor + margins - ds, torch.zeros_like(ds)).mean()
        torch.testing.assert_close(adaptive_reranking_loss(x, q, y), ref)

    def test_both_adapters_receive_gradients_and_only_residuals_train(self):
        x, y, _, positive, negative, _, _ = fixture()
        m = BinarySimNL(x, y[:, 0], positive[0], negative[0], SimNLConfig())
        initial = m.support_features.clone()
        loss, parts = m.loss()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"cross_entropy", "reranking", "text_consistency"})
        self.assertEqual(sum(p.numel() for p in m.parameters()), 10 * x.shape[1])
        for adapter in [m.positive_adapter, m.negative_adapter]:
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in adapter.parameters()), 0)
        self.assertFalse(m.support_features.requires_grad)
        torch.testing.assert_close(m.support_features, initial)

    def test_validation_features_never_enter_gradient_updates(self):
        x, y, _, positive, negative, _, _ = fixture()
        kwargs = dict(epochs=[0, 2], horizon=4, config=SimNLConfig(), device="cpu")
        a, _, _ = _train_snapshots(x[:12], y[:12, 0], positive[0], negative[0],
                                    validation_features=x[12:], **kwargs)
        b, _, _ = _train_snapshots(x[:12], y[:12, 0], positive[0], negative[0],
                                    validation_features=-x[12:], **kwargs)
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=0, rtol=0)


class SimNLProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_group_leakage_and_missing_training_class_rejected(self):
        x, y, logits, positive, negative, groups, folds = fixture()
        bad = [{"train": list(range(1, 20)), "validation": [0]},
               {"train": [0], "validation": list(range(1, 20))}]
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_folds(bad, groups, y)
        train = np.where(y[:, 0] == 0)[0]
        val = np.where(y[:, 0] == 1)[0]
        with self.assertRaisesRegex(ValueError, "training has one class"):
            validate_folds([{"train": train, "validation": val}], np.arange(20), y)

    def test_fit_selects_one_epoch_and_prediction_cannot_take_labels(self):
        x, y, logits, positive, negative, groups, folds = fixture()
        model = fit_simnl_ovr(x, y, logits, positive, negative, groups,
                             folds=folds, epoch_candidates=(0, 1, 3), device="cpu")
        self.assertIn(model.fit_info["chosen_epoch"], [0, 1, 3])
        self.assertEqual(model.fit_info["cosine_horizon"], 3)
        self.assertEqual(len(model.fit_info["oof_table"]), 3)
        self.assertEqual(model.fit_info["trainable_parameters"], 2 * 10 * 7)
        score = model.predict(x[:4], logits[:4])
        self.assertEqual(score.shape, (4, 2))
        np.testing.assert_array_equal(score, model.predict(x[:4], logits[:4] + 1000))
        self.assertNotIn("test_labels", inspect.signature(fit_simnl_ovr).parameters)
        self.assertNotIn("test_labels", inspect.signature(model.predict).parameters)
        with self.assertRaises(TypeError):
            model.predict(x[:4], logits[:4], test_labels=y[:4])

    def test_mismatched_queries_zero_features_or_invalid_fold_rejected(self):
        x, y, logits, positive, negative, groups, folds = fixture()
        with self.assertRaisesRegex(ValueError, "query shape mismatch"):
            fit_simnl_ovr(x, y, logits, positive[:1], negative, groups, folds=folds)
        x[0] = 0
        with self.assertRaisesRegex(ValueError, "zero feature"):
            fit_simnl_ovr(x, y, logits, positive, negative, groups, folds=folds)


if __name__ == "__main__":
    unittest.main()
