"""Real-checkpoint CPU tests for MedKLIP and MAVL token boundaries.

Run with the respective model's python_local.sh and --model MedKLIP or MAVL.
No fitting dataset or held-out labels are used by these numerical tests.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/baseline_token_classification"))
from common import residual_mlp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["MedKLIP", "MAVL"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    path = ROOT / f"baselines/{args.model}/local_setup/token_mlp_adapter.py"
    spec = importlib.util.spec_from_file_location("tested_token_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    openi = json.loads((ROOT / "RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json").read_text())
    chex = json.loads((ROOT / "data/direct_anchor_classification_20261118/chexpert/dataset.json").read_text())
    chest = json.loads((ROOT / "data/direct_anchor_classification_20261118/chestxray14/dataset.json").read_text())
    coverage = {"openi": module.class_coverage(openi["class_names"]),
                "chexpert": module.class_coverage(chex["class_names"]),
                "chestxray14": module.class_coverage(chest["class_names"])}
    assert coverage["openi"]["missing"] == ["Calcified Granuloma", "Granuloma"]
    assert coverage["chexpert"]["full_task_supported"]
    assert coverage["chestxray14"]["full_task_supported"]
    common16 = [name for name in openi["class_names"] if name not in coverage["openi"]["missing"]]
    assert len(common16) == 16 and module.class_coverage(common16)["full_task_supported"]

    backend = module.TokenBackend("cpu")
    try:
        backend.prepare_classes(openi["class_names"])
    except NotImplementedError as error:
        assert "Calcified Granuloma" in str(error) and "Granuloma" in str(error)
    else:
        raise AssertionError("Full OpenI18 must not be silently reduced")
    backend.prepare_classes(common16)
    assert backend.indices[0] == backend.native.classes.index("atelectasis")
    backend.prepare_classes(chex["class_names"])
    rows = json.loads(Path(chex["samples_file"]).read_text())
    images = [rows[0]["image"]]
    tokens = backend.encode_images(images)
    assert tokens.shape == (1, 196, 256) and not tokens.requires_grad
    assert not torch.is_inference(tokens)
    adapter = residual_mlp(dim=256, bottleneck=64)
    assert sum(p.numel() for p in adapter.parameters()) == 33088
    assert not any(p.requires_grad for p in backend.model.parameters())
    with torch.no_grad():
        native_logits = backend.native_scores(images)
        cached_logits = backend.native_scores_from_tokens(tokens)
        zero_logits = backend.score_tokens(tokens, adapter)
        native_probs = backend.native_predictions(images)
    torch.testing.assert_close(cached_logits, native_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(zero_logits, native_logits, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(cached_logits.sigmoid(), native_probs, atol=2e-6, rtol=2e-6)

    # First update: only the zero-initialized up projection has nonzero gradient.
    target = torch.linspace(-1, 1, native_logits.numel()).reshape_as(native_logits)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.01)
    loss = (backend.score_tokens(tokens, adapter) - target).square().mean()
    loss.backward()
    assert float(adapter.up.weight.grad.abs().sum()) > 0
    assert float(adapter.down.weight.grad.abs().sum()) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Independent author forward with its image_encoder output intercepted.
    # This checks all downstream operations with nonzero token modifications.
    original_encoder = backend.model.image_encoder
    pixels = backend._pixels(images)
    with torch.no_grad():
        adapted = adapter(tokens)
        if args.model == "MedKLIP":
            replacement = adapted
        else:
            _, global_feature = original_encoder(pixels)
            replacement = (adapted.transpose(1, 2).reshape(1, 256, 14, 14), global_feature)
        backend.model.image_encoder = lambda _: replacement
        try:
            author_logits = backend.model(pixels)
            if args.model == "MAVL":
                author_logits = author_logits[0]
            author_margin = (author_logits[..., 1] - author_logits[..., 0])[:, backend.indices]
        finally:
            del backend.model.image_encoder
        changed = backend.score_tokens(tokens, adapter)
    torch.testing.assert_close(changed, author_margin, atol=2e-5, rtol=2e-5)
    assert not torch.equal(changed, native_logits)

    (backend.score_tokens(tokens, adapter) - target).square().mean().backward()
    assert float(adapter.down.weight.grad.abs().sum()) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert all(p.grad is None for p in backend.model.parameters())
    with torch.no_grad():
        before_release = backend.score_tokens(tokens, adapter)
        backend.release_image_encoder()
        after_release = backend.score_tokens(tokens, adapter)
    torch.testing.assert_close(before_release, after_release, atol=0, rtol=0)
    print(json.dumps({"status": "passed", "model": args.model, "device": "cpu",
                      "token_shape": list(tokens.shape), "adapter_parameters": 33088,
                      "native_max_logit_error": float((cached_logits - native_logits).abs().max()),
                      "native_max_probability_error": float((cached_logits.sigmoid() - native_probs).abs().max()),
                      "nonzero_author_forward_max_logit_error": float((changed - author_margin).abs().max()),
                      "frozen_backbone_and_adapter_gradients": "passed",
                      "openi_common16": common16, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
