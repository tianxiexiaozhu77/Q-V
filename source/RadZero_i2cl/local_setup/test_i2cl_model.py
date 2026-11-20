#!/usr/bin/env python3
"""CPU checks of I2CL identity, gradient flow, immutable context, and reload.

Run from RadZero with:
    .venv/bin/python local_setup/test_i2cl_model.py

Uses two tiny real Dinov2Encoder blocks and the released RadZero scoring code;
does not load pretrained weights, download assets, or allocate GPU tensors.
"""

import importlib.util
import io
import sys
import unittest

from infer import MODEL_DIR, configure_offline

configure_offline()

import torch
from torch import nn
from transformers import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder

from i2cl_model import RadZeroI2CL


def load_released_loss():
    spec = importlib.util.spec_from_file_location(
        "radzero_i2cl_cpu_test_released_losses", MODEL_DIR / "losses.py"
    )
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.KeyPhraseAlignmentLoss


ReleasedLoss = load_released_loss()


class IdentityVision(nn.Module):
    def forward(self, pixels):
        return {"last_hidden_state": pixels}


class TinyAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layers = Dinov2Encoder(
            Dinov2Config(
                hidden_size=16, num_attention_heads=4, num_hidden_layers=2, mlp_ratio=2
            )
        )

    def forward(self, tokens):
        return self.transformer_layers(tokens)["last_hidden_state"]


class TinyRadZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = IdentityVision()
        self.align_transformer = TinyAlignment()
        self.text_model = nn.Embedding(16, 16)
        self.loss_fns = nn.ModuleDict(
            {"KeyPhraseAlignmentLoss": ReleasedLoss(hidden_dim=16)}
        )

    def forward_text_model(self, encoded):
        features = self.text_model(encoded["input_ids"]).mean(dim=1)
        return {
            "text_features_wo_l2_norm": features,
            "text_features": nn.functional.normalize(features, dim=-1),
        }


def tokenizer(prompts, **_kwargs):
    ids = torch.arange(len(prompts) * 2).reshape(len(prompts), 2)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class I2CLModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def make_wrapper(self, classes=2):
        base = TinyRadZero()
        prompts = [f"finding {index}" for index in range(classes)]
        # Construction must not leave inference tensors in a trainable path.
        with torch.inference_mode():
            wrapper = RadZeroI2CL(base, tokenizer, prompts)
        tokens = wrapper.encode_images(torch.randn(5, 5, 16))
        wrapper.build_context(tokens, batch_size=2)
        return base, wrapper, tokens, prompts

    def test_identity_matches_released_scoring_for_singletons_and_batches(self):
        for classes in (1, 3):
            base, wrapper, tokens, prompts = self.make_wrapper(classes)
            encoded = tokenizer(prompts)
            split_prompts = [
                {key: value[index:index + 1] for key, value in encoded.items()}
                for index in range(classes)
            ]
            loss = base.loss_fns["KeyPhraseAlignmentLoss"]
            for batch_size in (1, 2):
                with self.subTest(classes=classes, batch_size=batch_size):
                    batch = tokens[:batch_size]
                    with torch.no_grad():
                        reference = loss(
                            split_prompts, base.align_transformer(batch),
                            base.forward_text_model, ddp_gather=False,
                            need_attn_weights=True, compute_loss=False,
                        )
                        expected_logits = (
                            reference["t2i_logits"].reshape(classes, batch_size).T
                            / loss.loss_temperature.exp()
                        )
                        expected_maps = torch.stack(
                            reference["t2i_attn_weights"]
                        ).mean(dim=0)[:, :, 1:].reshape(batch_size, classes, 2, 2)
                        with wrapper.identity_coefficients():
                            actual = wrapper.forward_tokens(batch, return_maps=True)
                    torch.testing.assert_close(actual["logits"], expected_logits, atol=0, rtol=0)
                    torch.testing.assert_close(
                        actual["similarity_scores"], expected_maps, atol=0, rtol=0
                    )

    def test_only_coefficients_train_and_serialized_state_restores_predictions(self):
        base, wrapper, tokens, _ = self.make_wrapper()
        self.assertEqual(wrapper.parameter_counts()["trainable"], 8)
        self.assertFalse(wrapper.text_features.is_inference())
        self.assertFalse(wrapper.context_vectors.is_inference())
        context = wrapper.context_vectors.clone()
        before_weights = [parameter.detach().clone() for parameter in base.parameters()]
        before_coefficients = wrapper.coefficients.detach().clone()
        wrapper.train()
        self.assertFalse(base.training)
        optimizer = torch.optim.AdamW(wrapper.trainable_parameters(), lr=0.01)
        labels = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.], [1., 0.]])
        output = wrapper.forward_tokens(tokens, training_noise=True)
        loss = nn.functional.binary_cross_entropy_with_logits(output["logits"], labels)
        loss.backward()
        gradient = wrapper.coefficients.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs() > 0).all())
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before_coefficients, wrapper.coefficients))
        torch.testing.assert_close(wrapper.context_vectors, context, atol=0, rtol=0)
        for before, after in zip(before_weights, base.parameters()):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        wrapper.eval()
        with torch.no_grad():
            expected = wrapper.forward_tokens(tokens, training_noise=True)["logits"]
            repeated = wrapper.forward_tokens(tokens, training_noise=True)["logits"]
        torch.testing.assert_close(expected, repeated, atol=0, rtol=0)
        # State must not alias the live tensors, including on CPU.
        state = wrapper.adaptation_state()
        with torch.no_grad():
            wrapper.coefficients.zero_()
            wrapper.context_vectors.zero_()
        self.assertGreater(float(state["coefficients"].abs().sum()), 0)
        self.assertGreater(float(state["context_vectors"].abs().sum()), 0)
        serialized = io.BytesIO()
        torch.save(state, serialized)
        serialized.seek(0)
        wrapper.load_adaptation_state(torch.load(serialized, map_location="cpu"))
        with torch.no_grad():
            restored = wrapper.forward_tokens(tokens)["logits"]
        torch.testing.assert_close(restored, expected, atol=0, rtol=0)
        for _, _, module in wrapper._sites():
            self.assertFalse(module._forward_hooks)

    def test_context_averages_all_examples_and_hooks_are_removed_on_failure(self):
        base, wrapper, tokens, _ = self.make_wrapper()
        batched = wrapper.context_vectors.clone()
        single = wrapper.build_context(tokens, batch_size=1)
        torch.testing.assert_close(single, batched, atol=2e-6, rtol=1e-5)
        self.assertEqual(wrapper.context_sample_count, len(tokens))
        final_mlp = base.align_transformer.transformer_layers.layer[-1].mlp
        original_forward = final_mlp.forward

        def fail(_tokens):
            raise RuntimeError("intentional interrupted forward")

        final_mlp.forward = fail
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional interrupted"):
                wrapper.forward_tokens(tokens)
        finally:
            final_mlp.forward = original_forward
        for _, _, module in wrapper._sites():
            self.assertFalse(module._forward_hooks)
        with torch.no_grad():
            self.assertTrue(torch.isfinite(wrapper.forward_tokens(tokens)["logits"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
