#!/usr/bin/env python3
"""CPU checks for support-derived positive weighting and real coefficient updates."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_i2cl_model import TinyRadZero, tokenizer
from i2cl_model import RadZeroI2CL
from train_i2cl import (
    calibrate, coefficient_identity_penalty, evaluate_support_loss,
    make_support_criterion, regularization_metadata,
)


class SupportLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        # Two classes with different frequencies; first/last batches can contain
        # just positives/negatives without making the fixed weights undefined.
        self.targets = torch.tensor([
            [1., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 0.],
        ])

    def test_positive_terms_and_gradients_have_class_specific_weights(self):
        criterion, metadata = make_support_criterion(self.targets)
        self.assertEqual(metadata["positive_counts"], [1, 4])
        self.assertEqual(metadata["negative_counts"], [4, 1])
        self.assertEqual(metadata["pos_weight"], [4., .25])
        logits = torch.zeros_like(self.targets, requires_grad=True)
        loss = criterion(logits, self.targets)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        # At z=0: positive gradients are -weight/2, negatives +1/2;
        # mean reduction divides by all ten target entries.
        expected = torch.tensor([
            [-.2, -.0125], [.05, -.0125], [.05, -.0125],
            [.05, -.0125], [.05, .05],
        ])
        torch.testing.assert_close(logits.grad, expected)
        self.assertTrue(torch.isfinite(criterion(torch.zeros(1, 2), self.targets[:1])))
        self.assertTrue(torch.isfinite(criterion(torch.zeros(1, 2), self.targets[-1:])))
        torch.testing.assert_close(criterion.pos_weight, torch.tensor([4., .25]))

    def test_five_positive_five_negative_is_equivalent_to_unweighted_bce(self):
        targets = torch.tensor([[1.]] * 5 + [[0.]] * 5)
        criterion, metadata = make_support_criterion(targets)
        self.assertEqual(metadata["pos_weight"], [1.])
        logits = torch.linspace(-2, 2, 10).reshape(-1, 1)
        torch.testing.assert_close(
            criterion(logits, targets),
            torch.nn.functional.binary_cross_entropy_with_logits(logits, targets),
            atol=0, rtol=0,
        )

    def test_missing_states_or_nonbinary_support_fail_before_training(self):
        for targets in (
            torch.zeros(3, 1), torch.ones(3, 1),
            torch.tensor([[1., 0.], [0., .5]]),
            torch.tensor([[1.], [float("nan")]]), torch.empty(0, 2),
        ):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    make_support_criterion(targets)

    def test_support_loss_uses_fixed_weights_and_includes_last_short_batch(self):
        criterion, _ = make_support_criterion(self.targets)
        logits = torch.randn_like(self.targets)

        class FixedScores(torch.nn.Module):
            def forward_tokens(self, values):
                return {"logits": values}

        # Independent stable weighted-BCE expression, averaged over B x C.
        expected = (
            self.targets * torch.tensor([4., .25]) * torch.nn.functional.softplus(-logits)
            + (1 - self.targets) * torch.nn.functional.softplus(logits)
        ).mean().item()
        for batch_size in (1, 2, 5):
            with self.subTest(batch_size=batch_size):
                actual = evaluate_support_loss(
                    FixedScores(), logits, self.targets, batch_size, criterion
                )
                self.assertAlmostEqual(actual, expected, places=6)

    def test_penalty_is_zero_at_identity_and_gradient_points_towards_identity(self):
        coefficients = torch.tensor([[[0., 1.]] * 2] * 2, requires_grad=True)
        loss = coefficient_identity_penalty(coefficients)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        torch.testing.assert_close(coefficients.grad, torch.zeros_like(coefficients))

        # Four alphas of 2 and four betas of 4: mean(2^2) + mean(3^2) = 13.
        coefficients = torch.tensor([[[2., 4.]] * 2] * 2, requires_grad=True)
        loss = coefficient_identity_penalty(coefficients)
        self.assertEqual(loss.item(), 13.)
        loss.backward()
        expected_gradient = torch.tensor([[[1., 1.5]] * 2] * 2)
        torch.testing.assert_close(coefficients.grad, expected_gradient)
        updated = coefficients.detach() - .1 * coefficients.grad
        self.assertLess(coefficient_identity_penalty(updated).item(), 13.)

    def test_regularization_rejects_invalid_strength(self):
        for value in (-.01, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                regularization_metadata(value)
        self.assertEqual(regularization_metadata(0.)["lambda"], 0.)

    def test_training_gradient_includes_only_the_requested_extra_penalty(self):
        base = TinyRadZero()
        wrapper = RadZeroI2CL(base, tokenizer, ["finding 0", "finding 1"], noise_scale=0.)
        tokens = wrapper.encode_images(torch.randn(5, 5, 16))
        wrapper.build_context(tokens, batch_size=2)
        initial = wrapper.coefficients.detach().clone()
        gradients = []
        for strength in (0., .01):
            candidate = copy.deepcopy(wrapper)
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(
                    dataset="synthetic", grad_batch_size=2, learning_rate=.01,
                    weight_decay=.001, lambda_reg=strength, seed=42, epochs=1,
                    output=Path(directory),
                )
                result = calibrate(candidate, candidate.model, tokens, self.targets.numpy(), args)
                gradients.append(torch.tensor(result["first_gradient"]))
        expected_extra = torch.empty_like(initial)
        expected_extra[..., 0] = .01 * 2 * initial[..., 0] / 4
        expected_extra[..., 1] = .01 * 2 * (initial[..., 1] - 1) / 4
        torch.testing.assert_close(
            gradients[1] - gradients[0], expected_extra, atol=1e-7, rtol=1e-4,
        )

    def test_calibration_weights_once_and_updates_only_eight_coefficients(self):
        base = TinyRadZero()
        wrapper = RadZeroI2CL(base, tokenizer, ["finding 0", "finding 1"])
        tokens = wrapper.encode_images(torch.randn(5, 5, 16))
        wrapper.build_context(tokens, batch_size=2)
        context = wrapper.context_vectors.clone()
        coefficients = wrapper.coefficients.detach().clone()
        original = {name: value.clone() for name, value in base.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                dataset="synthetic", grad_batch_size=2, learning_rate=.01,
                weight_decay=.001, lambda_reg=.01, seed=42, epochs=2, output=Path(directory),
            )
            with patch("train_i2cl.make_support_criterion", wraps=make_support_criterion) as factory:
                result = calibrate(wrapper, base, tokens, self.targets.numpy(), args)
            self.assertEqual(factory.call_count, 1)
            history = json.loads((Path(directory) / "train_history.json").read_text())
            for entry in history:
                self.assertAlmostEqual(
                    entry["train_regularization_loss"],
                    .01 * entry["train_coefficient_penalty"], places=6,
                )
                self.assertAlmostEqual(
                    entry["train_total_loss_with_noise"],
                    entry["train_bce_with_noise"] + entry["train_regularization_loss"], places=6,
                )
        self.assertEqual(result["regularization"], regularization_metadata(.01))
        self.assertEqual(result["loss_weighting"]["pos_weight"], [4., .25])
        self.assertEqual(result["optimizer_steps"], 6)
        self.assertEqual(result["examples_per_epoch"], 5)
        self.assertTrue(result["all_eight_coefficients_received_nonzero_finite_gradients"])
        self.assertFalse(torch.equal(coefficients, wrapper.coefficients))
        torch.testing.assert_close(context, wrapper.context_vectors, atol=0, rtol=0)
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        criterion, _ = make_support_criterion(self.targets)
        with torch.no_grad():
            expected = criterion(wrapper.forward_tokens(tokens)["logits"], self.targets).item()
        self.assertAlmostEqual(result["final_support_bce_without_noise"], expected, places=6)
        final_penalty = coefficient_identity_penalty(wrapper.coefficients).item()
        self.assertAlmostEqual(result["final_coefficient_penalty"], final_penalty, places=6)
        self.assertAlmostEqual(
            result["final_support_total_loss_without_noise"],
            expected + .01 * final_penalty, places=6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
