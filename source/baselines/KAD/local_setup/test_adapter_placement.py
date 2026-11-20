"""Real-checkpoint CPU audit of matched pre/post KAD adapter placement."""
from pathlib import Path
import hashlib
import json
import sys

import torch
from torch import nn

from token_mlp_adapter import TokenBackend

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import residual_mlp, query_hashes


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    paths = [str(ROOT / "data/alpharad/ChestXDet10/test_data" / name)
             for name in ("36199.png", "36302.png")]
    backend = TokenBackend("cpu")
    backend.prepare_classes(["Atelectasis", "Pneumothorax"])
    tokens = backend.encode_images(paths)
    native = backend.native_scores(paths)
    probabilities = backend.native_predictions(paths)
    before_hashes = query_hashes(backend)
    contexts = backend.encode_contexts(tokens)
    assert contexts.shape == (2, 2, 768)
    assert not contexts.requires_grad and not torch.is_inference(contexts)

    default = backend.score_tokens(tokens)
    torch.testing.assert_close(default, native, rtol=0, atol=0)
    torch.testing.assert_close(default.sigmoid(), probabilities, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(backend.score_features(contexts), default, rtol=0, atol=0)
    report = {
        "status": "passed", "device": "cpu", "token_shape": list(tokens.shape),
        "context_shape": list(contexts.shape), "parameter_count": 99136,
        "native_default_max_abs_error": float((default - native).abs().max()),
        "native_probability_max_abs_error": float((default.sigmoid() - probabilities).abs().max()),
        "context_cache_max_abs_error": float((backend.score_features(contexts) - default).abs().max()),
        "positions": {},
    }
    targets = torch.eye(2)
    for position in ("pre", "post"):
        backend.configure_adapter_position(position)
        torch.manual_seed(73)
        adapter = residual_mlp(768, 64)
        assert sum(p.numel() for p in adapter.parameters()) == 99136
        zero = backend.score_tokens(tokens, adapter, adapter_position=position)
        torch.testing.assert_close(zero, native, rtol=0, atol=0)
        # Nonzero upper weights exercise gradients through both MLP layers.
        nn.init.normal_(adapter.up.weight, std=0.001)
        nn.init.normal_(adapter.up.bias, std=0.001)
        predicted = backend.score_tokens(tokens, adapter, adapter_position=position)
        if position == "pre":
            native_pairs = backend.model(adapter(tokens), backend.query)
            # The old public call continues to mean pre, independent of metadata.
            torch.testing.assert_close(backend.score_tokens(tokens, adapter), predicted, rtol=0, atol=0)
        else:
            # Independent native-forward oracle: replace only the original
            # classifier's input; all native decoder code executes unchanged.
            handle = backend.model.mlp_head.register_forward_pre_hook(
                lambda module, inputs: (adapter(inputs[0]),))
            try:
                native_pairs = backend.model(tokens, backend.query)
            finally:
                handle.remove()
            cached = backend.score_features(contexts, adapter)
            torch.testing.assert_close(cached, predicted, rtol=0, atol=0)
            # Explicit position requests must not alter the legacy default.
            default_after_config = backend.score_tokens(tokens, adapter)
            manual_pre = backend.model(adapter(tokens), backend.query)
            torch.testing.assert_close(default_after_config, manual_pre[..., 1] - manual_pre[..., 0], rtol=0, atol=0)
        oracle = native_pairs[..., 1] - native_pairs[..., 0]
        torch.testing.assert_close(predicted, oracle, rtol=0, atol=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, targets)
        loss.backward()
        gradients = {name: float(parameter.grad.norm()) for name, parameter in adapter.named_parameters()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
        assert gradients["down.weight"] > 0 and gradients["up.weight"] > 0
        assert float((predicted - native).abs().max()) > 0
        for model in (backend.native.image_encoder, backend.native.text_encoder, backend.model):
            assert not model.training
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        report["positions"][position] = {
            "identity_max_abs_error": float((zero - native).abs().max()),
            "native_injection_max_abs_error": float((predicted - oracle).abs().max()),
            "nonzero_score_change": float((predicted - native).abs().max()),
            "gradient_norms": gradients,
            "adapter_insertion": backend.metadata["adapter_insertion"],
            "adapter_sharing": backend.metadata["adapter_sharing"],
        }

    assert before_hashes == query_hashes(backend)
    for invalid in ("middle", "PRE", None):
        try:
            backend.score_tokens(tokens, adapter_position=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid adapter position was accepted")
    report["frozen_backbone"] = True
    report["query_hashes_unchanged"] = before_hashes
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(__file__).with_name("token_mlp_adapter.py")]
    }
    target = Path(__file__).with_name("adapter_placement_cpu_validation.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
