"""CPU tests with the tiny real DINO alignment and released matching code."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_i2cl_model import TinyRadZero, tokenizer
from i2cl_model import RadZeroI2CL
from overnight_i2cl_variants import (
    VARIANTS, ConservativeI2CL, author_lr_factor, calibrate_variant,
    require_no_hooks, schedule_description, score_variants_from_pixels,
    strict_full_baseline_check, verify_export_roundtrip, load_verified_support_tokens,
)
import train_i2cl as original_training
from overnight_cache_methods import sha256


class ConservativeVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyRadZero()
        self.tokens = torch.randn(6, 5, 16)
        self.labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
        self.prompts = ["finding a", "finding b"]

    def adapter(self, variant):
        config = VARIANTS[variant]
        adapter = ConservativeI2CL(
            self.model, tokenizer, self.prompts, sites=config["sites"],
            learn_beta=config["learn_beta"], noise_scale=.001)
        adapter.build_context(self.tokens, batch_size=2)
        return adapter

    def test_author_schedule_exact_formula_and_actual_optimizer_order(self):
        description = schedule_description(100, 10, 20, 2)
        self.assertEqual(description["warmup_steps"], 5)
        self.assertEqual(author_lr_factor(0, 100, 5), 0)
        self.assertEqual(author_lr_factor(5, 100, 5), 1)
        self.assertAlmostEqual(author_lr_factor(6, 100, 5), .5*(1+np.cos(np.pi*6/100)))
        self.assertEqual(author_lr_factor(100, 100, 5), 0)
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=.003)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: author_lr_factor(step, 100, 5))
        used = []
        for step in range(100):
            optimizer.zero_grad()
            parameter.sum().backward()
            used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        np.testing.assert_allclose(used, [.003*author_lr_factor(step, 100, 5)
                                          for step in range(100)], atol=1e-15)

    def test_real_parameter_counts_identity_and_no_hooks_leak(self):
        counts = {"A": 8, "B": 8, "C": 4, "D": 4, "E": 4, "F": 2}
        for name, count in counts.items():
            with self.subTest(name=name):
                adapter = self.adapter(name).eval()
                self.assertEqual(sum(p.numel() for p in adapter.parameters() if p.requires_grad), count)
                self.assertFalse(adapter.coefficients.requires_grad)
                with torch.no_grad():
                    actual = adapter.forward_tokens(self.tokens[:2], return_maps=True)
                    expected = adapter.forward_tokens(self.tokens[:2], inject=False, return_maps=True)
                for key in actual:
                    torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
                self.assertTrue(adapter.assert_inactive_identity())
                require_no_hooks(adapter)

    def test_frozen_beta_immune_to_adamw_and_gradients_reach_active_parameters(self):
        adapter = self.adapter("C")
        optimizer = torch.optim.AdamW([adapter.active_deltas], lr=.1, weight_decay=.5)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        adapter.train()
        result = adapter.forward_tokens(self.tokens, training_noise=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            result["logits"], torch.tensor(self.labels, dtype=torch.float32))
        loss.backward()
        self.assertTrue((adapter.active_deltas.grad != 0).all())
        optimizer.step()
        torch.testing.assert_close(adapter.effective_coefficients()[..., 1],
                                   torch.ones(2, 2), atol=0, rtol=0)
        self.assertIsNone(adapter.coefficients.grad)
        for old, parameter in zip(before, self.model.parameters()):
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(old, parameter, atol=0, rtol=0)

    def test_lastblock_inactive_hook_neither_modifies_nor_adds_noise(self):
        adapter = self.adapter("F")
        activation = torch.randn(2, 5, 16)
        outputs = (activation, torch.tensor([7]))
        with patch("torch.randn_like", side_effect=AssertionError("Inactive noise called")):
            result = adapter._injection_hook(0, 0, True)(None, (), outputs)
        self.assertIs(result, outputs)
        with torch.no_grad():
            adapter.active_deltas.fill_(.2)
        adapter.assert_inactive_identity()
        expected = torch.tensor([[0., 1.], [0., 1.]])
        torch.testing.assert_close(adapter.effective_coefficients()[0], expected, atol=0, rtol=0)

    def test_export_reload_into_new_original_wrapper_is_equivalent(self):
        adapter = self.adapter("F")
        with torch.no_grad():
            adapter.active_deltas.copy_(torch.tensor([.12, -.17]))
        with tempfile.TemporaryDirectory() as directory:
            check = verify_export_roundtrip(
                adapter, tokenizer, self.tokens, Path(directory) / "state.pt")
        self.assertTrue(check["passed"])
        self.assertEqual(check["reloaded_class"], "original RadZeroI2CL")

    def test_variant_load_rejects_nonidentity_fixed_coordinate(self):
        adapter = self.adapter("F")
        exported = adapter.adaptation_state()
        exported["coefficients"][0, 0, 0] = .1
        with self.assertRaises(ValueError):
            adapter.load_adaptation_state(exported)

    def test_shared_test_encoder_only_runs_once_for_all_variants(self):
        base = RadZeroI2CL(self.model, tokenizer, self.prompts)
        adapters = {name: self.adapter(name).eval() for name in ("A", "C", "F")}
        with patch.object(self.model.vision_model, "forward",
                          wraps=self.model.vision_model.forward) as encode:
            baseline, logits = score_variants_from_pixels(base, adapters, self.tokens[:2])
        self.assertEqual(encode.call_count, 1)
        for values in logits.values():
            torch.testing.assert_close(values, baseline, atol=0, rtol=0)

    def test_training_only_updates_selected_scalars_and_records_truthful_schedule(self):
        adapter = self.adapter("F")
        context = adapter.context_vectors.clone()
        args = SimpleNamespace(
            deadline=None, seed=42, grad_batch_size=2, epochs=3,
            weight_decay=.001, dataset="Toy")
        config = {"name": "F", **VARIANTS["F"]}
        record = calibrate_variant(adapter, self.tokens, self.labels, config, args)
        self.assertEqual(record["active_parameter_count"], 2)
        self.assertEqual(record["optimizer_steps"], 9)
        self.assertEqual(len(record["learning_rates_used"]), 9)
        self.assertEqual(record["learning_rates_used"][0], 0.)
        self.assertEqual(record["initial_coefficients"], [[[0., 1.], [0., 1.]],
                                                         [[0., 1.], [0., 1.]]])
        self.assertTrue(record["all_active_parameters_received_nonzero_finite_gradients"])
        self.assertTrue(record["inactive_coefficients_exact_identity"])
        torch.testing.assert_close(context, adapter.context_vectors, atol=0, rtol=0)
        self.assertTrue((adapter.active_deltas != 0).all())
        require_no_hooks(adapter)

    def test_baseline_mismatch_is_rejected(self):
        actual = np.asarray([[.1, .3]], dtype=np.float32)
        self.assertEqual(strict_full_baseline_check(actual, actual.copy()), 0)
        with self.assertRaises(ValueError):
            strict_full_baseline_check(actual + .001, actual)

    def test_support_cache_requires_matching_model_source_split_identity_and_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, splits, model_dir = root / "run", root / "splits", root / "model"
            directory.mkdir()
            (splits / "Toy").mkdir(parents=True)
            model_dir.mkdir()
            for filename in ("config.json", "preprocessor_config.json"):
                (model_dir / filename).write_text("{}")
            support_csv = splits / "Toy" / "support.csv"
            support_csv.write_text("support fixture")
            source = root / "producer.py"
            source.write_text("original producer\n")
            support = {
                "image_paths": ["a.png", "b.png"], "labels": np.asarray([[1], [0]]),
                "ids": ["a", "b"], "groups": ["a", "b"],
                "class_names": ["finding"], "prompts": ["There is finding"],
            }
            support_identity = original_training.fingerprint(original_training.spec_identity(support))
            tokens_path = directory / "support_tokens.npy"
            np.save(tokens_path, np.zeros((2, 1370, 768), dtype=np.float32), allow_pickle=False)
            identities = {key: support_identity for key in
                          ("support_identity", "context_support_identity", "calibration_support_identity")}
            token_metadata = {
                **identities, "token_path": str(tokens_path),
                "token_sha256": sha256(tokens_path), "shape": [2, 1370, 768],
                "dtype": "float32", "token_stage": "frozen DINO output BEFORE the alignment transformer",
            }
            (directory / "support_cache.json").write_text(json.dumps(token_metadata))
            config = {
                **identities, "checkpoint": {"hash": "fake"},
                "class_names": support["class_names"], "prompts": support["prompts"],
                "source_sha256": {str(source): sha256(source)},
                "model_config_sha256": sha256(model_dir / "config.json"),
                "preprocessor_sha256": sha256(model_dir / "preprocessor_config.json"),
                "split_sha256": {str(support_csv): sha256(support_csv)},
            }
            (directory / "run_config.json").write_text(json.dumps(config))
            frozen = {"passed": True, "before_sha256": "model-digest", "after_sha256": "model-digest"}
            (directory / "frozen_model_check.json").write_text(json.dumps(frozen))
            cache = SimpleNamespace(dataset="Toy", split_dir=splits,
                                    manifest={"checkpoint": {"hash": "fake"}})
            with patch("overnight_i2cl_variants.MODEL_DIR", model_dir), \
                 patch("overnight_i2cl_variants.checkpoint_identity", return_value={"hash": "fake"}):
                loaded, audit = load_verified_support_tokens(
                    directory, support, cache, "cpu", expected_model_digest="model-digest")
                self.assertEqual(tuple(loaded.shape), (2, 1370, 768))
                self.assertTrue(audit["reused"])
                with self.assertRaisesRegex(ValueError, "model tensor digest"):
                    load_verified_support_tokens(
                        directory, support, cache, "cpu", expected_model_digest="other")
                source.write_text("modified producer\n")
                with self.assertRaisesRegex(ValueError, "generating source changed"):
                    load_verified_support_tokens(directory, support, cache, "cpu")
                source.write_text("original producer\n")
                token_metadata["token_sha256"] = "corrupted"
                (directory / "support_cache.json").write_text(json.dumps(token_metadata))
                with self.assertRaisesRegex(ValueError, "content checksum"):
                    load_verified_support_tokens(directory, support, cache, "cpu")


if __name__ == "__main__":
    unittest.main()
