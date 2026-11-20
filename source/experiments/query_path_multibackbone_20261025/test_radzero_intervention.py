"""CPU native-code/weight checks for RadZero Original/R/Q interventions.

Run with ``CUDA_VISIBLE_DEVICES='' bash RadZero_i2cl/local_setup/python_local.sh
experiments/query_path_multibackbone_20261025/test_radzero_intervention.py``.
No image encoder is loaded and no GPU memory is allocated.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RadZero_i2cl/local_setup"))
from infer import MODEL_DIR, configure_offline

configure_offline()
from overnight_query_refinement import QueryRefinementMatcher
from radzero_intervention import (ARMS, RadZeroPathMatcher, audit_native_equivalence,
                                 evaluate_interventions, match_intervention, prepare_vision)


def load_released_loss():
    name = "radzero_path_test_actual_released_loss"
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / "losses.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.KeyPhraseAlignmentLoss


ReleasedLoss = load_released_loss()


class MinimalModel(nn.Module):
    def __init__(self, hidden=16, **kwargs):
        super().__init__()
        self.loss_fns = nn.ModuleDict({"KeyPhraseAlignmentLoss": ReleasedLoss(hidden_dim=hidden, **kwargs)})
        self.vision_calls = 0

    def forward_vision_model(self, pixels):
        self.vision_calls += 1
        return {"vision_tokens": pixels}


class NativeInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20261025)

    def queries(self, classes, hidden):
        original = torch.randn(classes, hidden)
        refined = F.normalize(F.normalize(original, dim=-1)
                              + .3 * F.normalize(torch.randn_like(original), dim=-1), dim=-1)
        return original, refined

    def test_released_scorer_and_existing_matcher_agree_for_singletons_and_native_options(self):
        for classes in (1, 3):
            for batch in (1, 2):
                for include_cls in (False, True):
                    for use_ln in (False, True):
                        for temperature in (None, .2):
                            with self.subTest(classes=classes, batch=batch, cls=include_cls,
                                              ln=use_ln, attention_temperature=temperature):
                                model = MinimalModel(use_vision_cls_token=include_cls,
                                    use_layer_norm=use_ln, attn_temperature=temperature).eval()
                                original, refined = self.queries(classes, 16)
                                matcher = QueryRefinementMatcher(model, torch.stack([original, refined]), [0., .3])
                                aligned = torch.randn(batch, 5, 16)
                                vision = prepare_vision(model, aligned)
                                torch.testing.assert_close(vision, matcher.prepare_vision(aligned), atol=0, rtol=0)
                                audit = audit_native_equivalence(model, vision, original, refined,
                                    reference_matcher=matcher, atol=2e-6, rtol=2e-6)
                                self.assertEqual(audit["status"], "passed")
                                outputs = evaluate_interventions(model, vision, original, refined)
                                self.assertEqual(tuple(outputs), ARMS)
                                self.assertEqual(outputs["query"]["logits"].shape, (batch, classes))
                                self.assertEqual(outputs["query"]["similarity_scores"].shape, (batch, classes, 2, 2))

    def test_q_minus_r_changes_only_pooling_when_readout_fixed(self):
        model = MinimalModel().eval()
        original, refined = self.queries(3, 16)
        vision = prepare_vision(model, torch.randn(2, 5, 16))
        arms = evaluate_interventions(model, vision, original, refined)
        for key in ("contexts", "attention_weights", "grounding_map"):
            torch.testing.assert_close(arms["original"][key], arms["readout_only"][key], atol=0, rtol=0)
        self.assertGreater(float((arms["query"]["contexts"] - arms["readout_only"]["contexts"]).abs().max()), 1e-5)
        qread = F.normalize(F.normalize(refined, dim=-1), dim=-1)
        for arm in ("query", "readout_only"):
            expected = (arms[arm]["contexts"] * qread[None]).sum(-1)
            expected /= model.loss_fns["KeyPhraseAlignmentLoss"].loss_temperature.exp()
            torch.testing.assert_close(arms[arm]["logits"], expected)

    def test_same_queries_collapse_all_arms_without_mutating_input_or_parameters(self):
        model = MinimalModel().eval()
        original, _ = self.queries(2, 16)
        pixels = torch.randn(3, 5, 16)
        before_pixels, before_q = pixels.clone(), original.clone()
        before_state = {key: value.clone() for key, value in model.state_dict().items()}
        wrapper = RadZeroPathMatcher(model, original, original)
        arms = wrapper(pixels)
        self.assertEqual(model.vision_calls, 1)
        for arm in ARMS:
            torch.testing.assert_close(arms[arm]["logits"], arms["original"]["logits"], atol=0, rtol=0)
            self.assertTrue(all(not value.requires_grad for value in arms[arm].values()))
        torch.testing.assert_close(pixels, before_pixels, atol=0, rtol=0)
        torch.testing.assert_close(original, before_q, atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[key], atol=0, rtol=0)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_audit_with_actual_checkpoint_matching_weights(self):
        from safetensors import safe_open
        config = json.loads((MODEL_DIR / "config.json").read_text())["loss"]["KeyPhraseAlignmentLoss"]
        model = MinimalModel(hidden=config["hidden_dim"], **{k:v for k,v in config.items() if k != "hidden_dim"})
        loss = model.loss_fns["KeyPhraseAlignmentLoss"]
        prefix = "loss_fns.KeyPhraseAlignmentLoss."
        with safe_open(MODEL_DIR / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            state = {key[len(prefix):]: checkpoint.get_tensor(key)
                     for key in checkpoint.keys() if key.startswith(prefix)}
        self.assertTrue(state, "Actual checkpoint must contain matching parameters")
        loss.load_state_dict(state, strict=True)
        original, refined = self.queries(3, config["hidden_dim"])
        matcher = QueryRefinementMatcher(model, torch.stack([original, refined]), [0., .3])
        prepared = prepare_vision(model, torch.randn(2, 17, config["hidden_dim"]))
        audit = audit_native_equivalence(model, prepared, original, refined, reference_matcher=matcher)
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(audit["native_source_sha256"])

    def test_rejects_incompatible_shapes_nonfinite_values_and_noncosine_scorer(self):
        model = MinimalModel().eval()
        q, refined = self.queries(3, 16)
        prepared = prepare_vision(model, torch.randn(2, 5, 16))
        with self.assertRaises(ValueError):
            match_intervention(model, prepared, q, refined[:1])
        with self.assertRaises(ValueError):
            prepare_vision(model, torch.full((2, 5, 16), float("nan")))
        with self.assertRaises(ValueError):
            match_intervention(model, prepared, q, torch.full_like(q, float("nan")))
        no_cls = MinimalModel(use_vision_cls_token=False)
        with self.assertRaises(ValueError):
            prepare_vision(no_cls, torch.randn(2, 1, 16))
        with self.assertRaises(ValueError):
            prepare_vision(MinimalModel(sim_op="dot"), torch.randn(2, 5, 16))


if __name__ == "__main__":
    unittest.main()
