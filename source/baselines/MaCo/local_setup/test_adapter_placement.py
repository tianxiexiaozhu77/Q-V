"""CPU-only placement contract tests without loading any model checkpoint.

Run from the medical workspace with:
  baselines/MaCo/local_setup/python_local.sh \
    baselines/MaCo/local_setup/test_adapter_placement.py
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import residual_mlp

spec = importlib.util.spec_from_file_location(
    "maco_placement_test_backend", Path(__file__).with_name("token_mlp_adapter.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MaCoPlacementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(390)
        torch.set_num_threads(2)
        self.backend = module.TokenBackend.__new__(module.TokenBackend)
        self.backend.class_names = ("Disease A", "Disease B", "Disease C")
        # Raw, unequal-length query vectors are essential: MaCo does not use
        # normalized queries for its final score.
        self.backend.positive = torch.randn(3, 768) * torch.tensor([.2, .8, 1.4])[:, None]
        self.backend.negative = torch.randn(3, 768) * torch.tensor([.7, .3, 1.1])[:, None]
        self.backend.queries = torch.cat((self.backend.positive, self.backend.negative))
        self.backend.attention_queries = F.normalize(self.backend.queries, dim=-1)
        self.tokens = torch.randn(2, 13, 768)
        self.adapter = residual_mlp(768, 64)

    def nonidentity_adapter(self):
        with torch.no_grad():
            self.adapter.up.weight.normal_(0, .04)
            self.adapter.up.bias.normal_(0, .02)
        return self.adapter

    def legacy_scores(self, tokens, adapter=None):
        # Frozen copy of the pre-placement implementation for bitwise parity.
        if adapter is not None:
            tokens = adapter(tokens)
        weights = torch.einsum("bnd,qd->bqn", F.normalize(tokens, dim=-1), self.backend.attention_queries).softmax(-1)
        contexts = torch.einsum("bqn,bnd->bqd", weights, tokens)
        scores = torch.einsum("bqd,qd->bq", contexts, self.backend.queries)
        return scores[:, :3] - scores[:, 3:]

    def test_default_is_bitwise_legacy_before_and_after_training(self):
        for adapter in (None, self.adapter):
            torch.testing.assert_close(
                self.backend.score_tokens(self.tokens, adapter),
                self.legacy_scores(self.tokens, adapter), rtol=0, atol=0)
        adapter = self.nonidentity_adapter()
        torch.testing.assert_close(
            self.backend.score_tokens(self.tokens, adapter),
            self.legacy_scores(self.tokens, adapter), rtol=0, atol=0)

    def test_identity_reproduces_each_pooling_baseline(self):
        native = self.tokens.mean(1) @ self.backend.positive.T - self.tokens.mean(1) @ self.backend.negative.T
        references = {"native_mean": native, "text_guided": self.legacy_scores(self.tokens)}
        for pooling, expected in references.items():
            for position in ("pre", "post"):
                with self.subTest(pooling=pooling, position=position):
                    actual = self.backend.score_tokens(self.tokens, self.adapter, position, pooling)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cached_post_features_equal_online_scores(self):
        adapter = self.nonidentity_adapter()
        for pooling, shape in (("native_mean", (2, 768)), ("text_guided", (2, 6, 768))):
            with self.subTest(pooling=pooling):
                features = self.backend.pooled_features(self.tokens, pooling).detach()
                self.assertEqual(tuple(features.shape), shape)
                torch.testing.assert_close(
                    self.backend.score_features(features, adapter, pooling),
                    self.backend.score_tokens(self.tokens, adapter, "post", pooling),
                    rtol=0, atol=0)

    def test_nonlinearity_makes_pre_and_post_distinct_for_both_poolings(self):
        adapter = self.nonidentity_adapter()
        for pooling in ("text_guided", "native_mean"):
            with self.subTest(pooling=pooling):
                before = self.backend.score_tokens(self.tokens, adapter, "pre", pooling)
                after = self.backend.score_tokens(self.tokens, adapter, "post", pooling)
                self.assertGreater((before - after).abs().max().item(), 1e-3)

    def test_raw_query_scale_is_retained_in_readout(self):
        adapter = self.nonidentity_adapter()
        for pooling in ("text_guided", "native_mean"):
            features = self.backend.pooled_features(self.tokens, pooling)
            scores = self.backend.score_features(features, adapter, pooling)
            originals = (self.backend.positive, self.backend.negative, self.backend.queries)
            self.backend.positive = originals[0] * 3
            self.backend.negative = originals[1] * 3
            self.backend.queries = originals[2] * 3
            scaled = self.backend.score_features(features, adapter, pooling)
            torch.testing.assert_close(scaled, scores * 3, rtol=1e-5, atol=1e-4)
            self.backend.positive, self.backend.negative, self.backend.queries = originals

    def test_gradients_reach_the_shared_mlp_in_all_four_paths(self):
        adapter = self.nonidentity_adapter()
        labels = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
        for pooling in ("text_guided", "native_mean"):
            for position in ("pre", "post"):
                with self.subTest(pooling=pooling, position=position):
                    adapter.zero_grad(set_to_none=True)
                    scores = self.backend.score_tokens(self.tokens, adapter, position, pooling)
                    F.binary_cross_entropy_with_logits(scores, labels).backward()
                    for name, param in adapter.named_parameters():
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all().item(), name)
                        self.assertGreater(param.grad.abs().sum().item(), 0, name)

    def test_identity_initialization_and_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.adapter.parameters()), 99136)
        torch.testing.assert_close(self.adapter(self.tokens), self.tokens, rtol=0, atol=0)
        scores = self.backend.score_tokens(self.tokens, self.adapter, "post")
        scores.square().mean().backward()
        self.assertGreater(self.adapter.up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.adapter.down.weight.grad.abs().sum().item(), 0)

    def test_unknown_position_or_pooling_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "position"):
            self.backend.score_tokens(self.tokens, adapter_position="mid")
        for function in (self.backend.score_tokens, self.backend.pooled_features, self.backend.score_features):
            with self.assertRaisesRegex(ValueError, "pooling"):
                function(self.tokens, pooling="unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
