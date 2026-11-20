"""CPU regression against the actual author's branch; no weights required.

Run with:
  bash AlphaRAD/local_setup/python_local.sh \
    experiments/query_path_intervention_20261024/test_intervention.py
"""

import ast
import itertools
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from intervention import (audit_native_equivalence, dense_query_map,
                          dense_query_residual_map, evaluate_interventions,
                          match_intervention)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "AlphaRAD/cxrclip/model/cxrclip_plus_dqn.py"


def load_native_branch():
    """Load only the unchanged native function, avoiding encoder dependencies."""
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "CXRClipPlusWithDQN")
    branch = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "multihead_radzero_classification_branch")
    namespace = {"torch": torch, "F": F}
    module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[branch.name]


class NativeFixture(nn.Module):
    multihead_radzero_classification_branch = load_native_branch()

    def __init__(self, heads=3, attend_cls=True, aggregation="mean"):
        super().__init__()
        self.num_radzero_heads = heads
        self.radzero_head_enable_cross_attend_cls_token = attend_cls
        self.radzero_logit_scale = nn.Parameter(torch.tensor(1.3))
        self.radzero_head_weights = nn.Parameter(torch.linspace(-1.2, 0.8, heads))
        self.use_learnable_head_weights = aggregation in ("weighted", "selected")
        self.head_index = heads - 1 if aggregation == "selected" else None
        self.radzero_head_enable_bce_loss = True
        self.radzero_head_enable_softmax_loss = False
        self.requires_grad_(False).eval()


def fixture(batch=2, classes=3, heads=3, patches=9, **kwargs):
    generator = torch.Generator().manual_seed(20261024)
    model = NativeFixture(heads=heads, **kwargs)
    dim = heads * 4
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    vision = {"image_cls_raw_features": [rand(batch, dim)],
              "image_patch_raw_features": [rand(batch, patches, dim)]}
    query = rand(classes, dim)
    direction = F.normalize(rand(classes, dim), dim=-1)
    refined = F.normalize(F.normalize(query, dim=-1) + 0.3 * direction, dim=-1)
    return model, vision, query, refined


class InterventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_and_q_match_actual_native_branch(self):
        # Singleton dimensions, selected-head precedence, nonuniform head
        # weighting, and CLS inclusion must all preserve the actual branch.
        for batch, classes, heads, cls, aggregation in itertools.product(
                (1, 2), (1, 3), (1, 3), (False, True),
                ("mean", "weighted", "selected")):
            with self.subTest(batch=batch, classes=classes, heads=heads,
                              cls=cls, aggregation=aggregation):
                args = fixture(batch, classes, heads, attend_cls=cls,
                               aggregation=aggregation)
                report = audit_native_equivalence(*args)
                self.assertEqual(report["status"], "passed")
                # The direct native-path operations are in identical order.
                for key, error in report["max_absolute_errors"].items():
                    if ".dense" not in key:
                        self.assertEqual(error, 0.0, key)

    def test_path_interventions_change_only_selected_path(self):
        args = fixture(aggregation="weighted")
        arms = evaluate_interventions(*args)
        for key in ("contexts", "grounding_map", "attention_weights"):
            self.assertTrue(torch.equal(arms["original"][key], arms["readout_only"][key]))
            self.assertTrue(torch.equal(arms["query"][key], arms["aggregation_only"][key]))
        self.assertFalse(torch.allclose(arms["original"]["contexts"], arms["query"]["contexts"]))
        self.assertFalse(torch.allclose(arms["original"]["logits"], arms["readout_only"]["logits"]))
        self.assertFalse(torch.allclose(arms["query"]["logits"], arms["aggregation_only"]["logits"]))
        for arm in arms.values():
            self.assertEqual(arm["similarity_scores"].shape, (2, 3, 3, 3))
            self.assertEqual(arm["grounding_map"].shape, (2, 3, 10))
            torch.testing.assert_close(arm["attention_weights"].sum(dim=2),
                                       torch.ones(2, 3, 3))

    def test_dense_residual_matches_q_without_context_pooling(self):
        for attend_cls in (False, True):
            model, vision, original, refined = fixture(attend_cls=attend_cls)
            native = model.multihead_radzero_classification_branch(
                vision, {"text_cls_raw_features": refined}, True)
            direct = dense_query_map(model, vision, refined)
            residual = dense_query_residual_map(model, vision, original, refined)
            for key in ("grounding_map", "grounding_per_head"):
                torch.testing.assert_close(direct[key], native[key], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(residual[key], native[key], atol=2e-6, rtol=2e-5)
            self.assertNotIn("contexts", direct)
            self.assertNotIn("attention_weights", residual)

    def test_zero_query_or_feature_subspaces_remain_native(self):
        model, vision, query, refined = fixture()
        query[0] = 0
        refined[1, :4] = 0
        vision["image_patch_raw_features"][0][:, 0] = 0
        report = audit_native_equivalence(model, vision, query, refined)
        self.assertEqual(report["status"], "passed")

    def test_no_parameter_or_input_mutation_and_no_grad_graph(self):
        model, vision, query, refined = fixture()
        query.requires_grad_(True)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        original_query = query.detach().clone()
        original_patch = vision["image_patch_raw_features"][0].clone()
        arms = evaluate_interventions(model, vision, query, refined)
        for arm in arms.values():
            self.assertTrue(all(not value.requires_grad for value in arm.values()))
        self.assertTrue(torch.equal(query, original_query))
        self.assertTrue(torch.equal(vision["image_patch_raw_features"][0], original_patch))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_non_square_maps_are_available_without_false_reshape(self):
        model, vision, query, refined = fixture(patches=7)
        result = match_intervention(model, vision, query, refined)
        self.assertEqual(result["spatial_map_flat"].shape, (2, 3, 7))
        self.assertNotIn("similarity_scores", result)

    def test_bad_shapes_and_nonfinite_inputs_fail(self):
        model, vision, query, refined = fixture()
        with self.assertRaises(ValueError):
            match_intervention(model, vision, query, refined[:, :-1])
        with self.assertRaises(ValueError):
            match_intervention(model, vision, query[:-1], refined)
        refined[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            match_intervention(model, vision, query, refined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
