"""CPU native-TQN/serialized-readout parity and full Direct-anchor gradients."""
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

from direct_anchor_adapter import DirectBackend, load_head

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import query_hashes, read_dataset
from experiments.direct_anchor_classification.model import DirectAnchor


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    backend = DirectBackend("cpu")
    report = {"model": "KAD", "device": "cpu", "datasets": {}}
    cache_root = ROOT / "experiments/baseline_token_classification/runs/20261119/cache/KAD"
    for dataset in ("openi", "chexpert", "chestxray14"):
        cache = json.loads((cache_root / dataset / "cache.json").read_text())
        _, rows = read_dataset(cache["dataset_path"])
        selected = [i for i, row in enumerate(rows) if row["split"] == "train"][:2]
        assert len(selected) == 2
        backend.prepare_classes(cache["class_names"])
        assert query_hashes(backend) == query_hashes(backend.token_backend)
        tokens = torch.from_numpy(np.array(np.load(cache["files"]["tokens.npy"]["path"], mmap_mode="r")[selected], copy=True))
        expected = torch.from_numpy(np.array(np.load(cache["files"]["native_logits.npy"]["path"], mmap_mode="r")[selected], copy=True))
        native = backend.score_tokens(tokens).detach()
        features = backend.aggregate_tokens(tokens)
        logits = backend.score_features(features)
        torch.testing.assert_close(logits, native, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(logits, expected, atol=3e-4, rtol=3e-4)
        with torch.no_grad():
            probability = backend.model(tokens, backend.query).softmax(-1)[..., 1]
            random_tokens = torch.randn_like(tokens)
            random_split = backend.score_features(backend.aggregate_tokens(random_tokens))
            random_native = backend.score_tokens(random_tokens)
        torch.testing.assert_close(logits.sigmoid(), probability, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(random_split, random_native, atol=1e-6, rtol=1e-6)

        adapter = DirectAnchor(len(cache["class_names"]), backend.dim, rank=8)
        torch.testing.assert_close(adapter(features), features, atol=0, rtol=0)
        torch.testing.assert_close(backend.score_features(adapter(features)), native, atol=1e-6, rtol=1e-6)
        torch.nn.init.normal_(adapter.up, std=.003)
        torch.nn.init.normal_(adapter.bias, std=.003)
        changed = adapter(features)
        score = backend.score_features(changed)
        def inject(module, inputs, output):
            return changed.detach().transpose(0, 1), output[1]
        hook = backend.model.decoder.register_forward_hook(inject)
        try:
            manual = backend.score_tokens(tokens).detach()
        finally:
            hook.remove()
        torch.testing.assert_close(score, manual, atol=1e-6, rtol=1e-6)
        stream = io.BytesIO()
        torch.save(backend.export_head(), stream)
        stream.seek(0)
        restored = load_head(torch.load(stream, map_location="cpu", weights_only=True), "cpu")
        restored.train()
        assert not any(m.training for m in restored.modules())
        torch.testing.assert_close(restored(changed), score, atol=0, rtol=0)
        labels = torch.arange(score.numel()).reshape_as(score).remainder(2).float()
        torch.nn.functional.binary_cross_entropy_with_logits(restored(changed), labels).backward()
        gradients = {name: float(value.grad.norm()) for name, value in adapter.named_parameters()}
        assert all(value > 0 and np.isfinite(value) for value in gradients.values())
        for module in (backend.model, backend.native.image_encoder, backend.native.text_encoder, backend.head, restored):
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
        report["datasets"][dataset] = {
            "train_indices": selected, "shape": list(features.shape), "original_query_alias_hash_match": True,
            "cpu_query_hash_matches_gpu_cache": query_hashes(backend) == cache["query_sha256"],
            "native_max_abs_error": float((logits-native).abs().max()),
            "old_cache_max_abs_error": float((logits-expected).abs().max()),
            "native_probability_max_abs_error": float((logits.sigmoid()-probability).abs().max()),
            "random_tokens_max_abs_error": float((random_split-random_native).abs().max()),
            "nonzero_injected_native_max_abs_error": float((score-manual).abs().max()),
            "gradient_norms": gradients, "head_roundtrip_exact": True,
        }
        print(json.dumps({"dataset": dataset, **report["datasets"][dataset]}), flush=True)
    report["status"] = "passed"
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
