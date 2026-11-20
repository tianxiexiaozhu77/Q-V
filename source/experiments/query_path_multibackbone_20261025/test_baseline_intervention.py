"""CPU checks using each repository's real native TransformerDecoder layers."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baseline_intervention import ROOT, VisualUpdateTape

NATIVE_DECODERS = {
    'CARZero': ('baselines/CARZero/CARZero/models/transformer_decoder.py', 'TransformerDecoderWoSelfAttenLayer'),
    'KAD': ('baselines/KAD/A3_CLIP/models/transformer_decoder.py', 'TransformerDecoderLayer'),
    'MedKLIP': ('baselines/MedKLIP/Sample_zero-shot_Classification_CXR14/models/transformer.py', 'TransformerDecoderLayer'),
    'MAVL': ('baselines/MAVL/Zero-shot_classification/models/transformer.py', 'TransformerDecoderLayer'),
}


def load_decoder(name):
    relative, layer_name = NATIVE_DECODERS[name]
    spec = importlib.util.spec_from_file_location('native_decoder_test_' + name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = getattr(module, layer_name)(16, 4, dim_feedforward=32, dropout=0.0, normalize_before=True)
    return module.TransformerDecoder(layer, 3, norm=torch.nn.LayerNorm(16)).eval().requires_grad_(False)


class NativeVisualUpdateTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(18)
        self.query = torch.randn(3, 2, 16)
        self.refined = self.query + .3 * torch.randn_like(self.query)
        self.memory = torch.randn(7, 2, 16)

    def test_all_native_decoders_reconstruct_and_isolate_visual_updates(self):
        for name in NATIVE_DECODERS:
            with self.subTest(model=name), torch.inference_mode():
                decoder = load_decoder(name)
                native_original = decoder(self.query, self.memory)[0]
                native_query = decoder(self.refined, self.memory)[0]
                tape = VisualUpdateTape(decoder)
                with tape.record():
                    original = decoder(self.query, self.memory)[0]
                with tape.replay():
                    reconstructed = decoder(self.query, self.memory)[0]
                with tape.replay():
                    fixed = decoder(self.refined, self.memory)[0]
                torch.testing.assert_close(original, native_original, rtol=0, atol=0)
                torch.testing.assert_close(reconstructed, native_original, rtol=0, atol=0)
                self.assertFalse(torch.allclose(fixed, native_query))
                self.assertFalse(torch.allclose(fixed, original))
                self.assertEqual(tape.audit()['replayed_layers'], 3)
                self.assertTrue(all(not module._forward_hooks for _, module in tape.modules))
                torch.testing.assert_close(decoder(self.refined, self.memory)[0], native_query, rtol=0, atol=0)

    def test_memory_mismatch_fails_and_removes_hooks(self):
        decoder = load_decoder('CARZero')
        tape = VisualUpdateTape(decoder)
        with torch.inference_mode(), tape.record():
            decoder(self.query, self.memory)
        with self.assertRaisesRegex(RuntimeError, 'different visual memory'):
            with torch.inference_mode(), tape.replay():
                decoder(self.refined, self.memory + 1)
        self.assertTrue(all(not module._forward_hooks for _, module in tape.modules))

    def test_exception_during_record_cleans_up(self):
        tape = VisualUpdateTape(load_decoder('KAD'))
        with self.assertRaisesRegex(ValueError, 'synthetic exception'):
            with tape.record():
                raise ValueError('synthetic exception')
        self.assertTrue(all(not module._forward_hooks for _, module in tape.modules))
        with self.assertRaisesRegex(RuntimeError, 'complete original-query'):
            with tape.replay():
                pass

    def test_missing_layers_are_rejected(self):
        tape = VisualUpdateTape(load_decoder('MedKLIP'))
        with self.assertRaisesRegex(RuntimeError, 'Not all native'):
            with tape.record():
                pass
        self.assertTrue(all(not module._forward_hooks for _, module in tape.modules))


if __name__ == '__main__':
    unittest.main()
