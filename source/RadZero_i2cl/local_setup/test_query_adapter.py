"""Portable query roundtrip, original identity and matching map semantics on CPU."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from query_adapter import FORMAT, QueryAdapter, save_bundle
from test_overnight_query_refinement import CountingRadZero


class PortableQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def metadata(self, eta, classes=2):
        return {"format": FORMAT, "dataset": "Fixture", "eta": eta,
                "class_names": [f"class{i}" for i in range(classes)],
                "prompts": [f"There is class{i}" for i in range(classes)]}

    def test_save_load_is_exact_and_contains_only_two_arrays(self):
        original = np.arange(32, dtype=np.float32).reshape(2, 16)
        refined = original / 32
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "query_adapter.npz"
            saved = save_bundle(path, original, refined, self.metadata(.3))
            loaded = QueryAdapter.load(folder)
            np.testing.assert_array_equal(saved.original, loaded.original)
            np.testing.assert_array_equal(saved.refined, loaded.refined)
            with np.load(path, allow_pickle=False) as values:
                self.assertEqual(set(values.files), {"original_queries", "refined_queries"})
            self.assertFalse(loaded.metadata["contains_support_images_or_labels"])
            self.assertFalse(loaded.original.flags.writeable)
            self.assertFalse(loaded.refined.flags.writeable)
            with self.assertRaises(FileExistsError):
                save_bundle(path, original, refined, self.metadata(.3))

    def test_query_checksum_tampering_and_support_metadata_rejected(self):
        q = np.ones((2, 16), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "query_adapter.npz"
            save_bundle(path, q, q, self.metadata(.3))
            np.savez(path, original_queries=q + 1, refined_queries=q)
            with self.assertRaisesRegex(ValueError, "checksum"):
                QueryAdapter.load(path)
        with self.assertRaisesRegex(ValueError, "must not be exported"):
            QueryAdapter(q, q, {**self.metadata(.3), "provenance": {"support_labels": [[1]]}})

    def test_eta_zero_loaded_adapter_exactly_preserves_original_match_and_maps(self):
        model = CountingRadZero().eval()
        query = torch.randn(2, 16)
        pixels = torch.randn(3, 5, 16)
        with tempfile.TemporaryDirectory() as folder:
            adapter = save_bundle(Path(folder) / "query_adapter.npz",
                                  query.numpy(), query.numpy(), self.metadata(0.))
            matcher = adapter.make_matcher(model)
            outputs = matcher(pixels, return_maps=True)
            self.assertEqual(model.vision_calls, 1)
            with torch.no_grad():
                vision = matcher.prepare_vision(model.align_transformer(pixels))
                expected, attention = model.loss_fns["KeyPhraseAlignmentLoss"].compute_t2i_logits(
                    query, vision, need_attn_weights=True)
                expected = expected.reshape(2, 3).T / matcher.loss.loss_temperature.exp()
            torch.testing.assert_close(outputs[0.]["logits"], expected, atol=0, rtol=0)
            torch.testing.assert_close(outputs[0.]["similarity_scores"],
                attention[0][:, :, 1:].reshape(3, 2, 2, 2), atol=0, rtol=0)

    def test_refined_query_loaded_maps_shape_and_single_visual_forward(self):
        for classes, batch in ((1, 1), (3, 2)):
            model = CountingRadZero().eval()
            original = torch.randn(classes, 16)
            refined = F.normalize(F.normalize(original, dim=-1) +
                                  .3 * F.normalize(torch.randn_like(original), dim=-1), dim=-1)
            with tempfile.TemporaryDirectory() as folder:
                adapter = save_bundle(Path(folder) / "query_adapter.npz", original.numpy(),
                                      refined.numpy(), self.metadata(.3, classes))
                matcher = QueryAdapter.load(adapter.path).make_matcher(model)
                outputs = matcher(torch.randn(batch, 5, 16), return_maps=True)
                self.assertEqual(model.vision_calls, 1)
                self.assertEqual(outputs[0.]["logits"].shape, (batch, classes))
                self.assertEqual(outputs[.3]["similarity_scores"].shape, (batch, classes, 2, 2))
                self.assertFalse(torch.equal(outputs[0.]["similarity_scores"],
                                             outputs[.3]["similarity_scores"]))
                self.assertTrue(all(not p.requires_grad for p in model.parameters()))
                self.assertTrue(all(not tensor.requires_grad for branch in outputs.values()
                                    for tensor in branch.values()))

    def test_general_cli_has_no_overnight_deadline(self):
        from eval_query_adapter import parse_args
        args = parse_args(["--datasets", "SIIM", "--output", "/tmp/query_adapter_fixture",
                           "--save-maps", "--map-limit", "3"])
        self.assertNotIn("deadline", vars(args))
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.map_limit, 3)


if __name__ == "__main__":
    unittest.main()
