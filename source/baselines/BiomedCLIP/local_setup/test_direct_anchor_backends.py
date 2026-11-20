"""CPU differential checks for the three lossless Direct-anchor feature splits.

The default fixtures exercise the unchanged production TokenBackend.score_tokens
with non-unit tokens and query magnitudes. --real-cache additionally loads the
real frozen checkpoint and checks two TRAIN cache rows; it never fits a model.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import types

import numpy as np
import torch
import torch.nn.functional as F

BASELINES = Path(__file__).resolve().parents[2]
MEDICAL = BASELINES.parent
MODELS = ("BiomedCLIP", "BioViL-T", "MaCo")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_hashes(backend):
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in vars(backend).items() if torch.is_tensor(value)
    }


def check(module, backend, tokens):
    old_scores = backend.token_backend.score_tokens(tokens)
    features = backend.aggregate_tokens(tokens)
    scores = backend.score_features(features)
    torch.testing.assert_close(scores, old_scores, rtol=0, atol=0)
    torch.testing.assert_close(backend.score_tokens(tokens), old_scores, rtol=0, atol=0)
    assert query_hashes(backend) == query_hashes(backend.token_backend)
    assert features.shape == (tokens.shape[0], len(backend.class_names), backend.dim)

    payload = backend.export_head()
    assert all(value.device.type == "cpu" for value in payload.values() if torch.is_tensor(value))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    head = module.load_head(torch.load(buffer, map_location="cpu", weights_only=True), device="cpu")
    assert head.dim == backend.dim and not head.training
    assert all(not parameter.requires_grad for parameter in head.parameters())
    assert all(not value.requires_grad for value in head.buffers())
    torch.testing.assert_close(head(features), scores, rtol=0, atol=0)

    train_features = features.detach().clone().requires_grad_(True)
    targets = (torch.arange(scores.numel()).reshape_as(scores) % 2).float()
    F.binary_cross_entropy_with_logits(head(train_features), targets).backward()
    assert train_features.grad is not None and torch.isfinite(train_features.grad).all()
    branch_gradients = [float(branch.norm()) for branch in train_features.grad.split(backend.token_dim, -1)]
    assert all(value > 0 for value in branch_gradients)
    perturbed = features + torch.randn_like(features) * 0.01
    change = float((head(perturbed) - scores).abs().max())
    assert change > 0

    backend.release_image_encoder()
    assert backend.token_backend.native is None
    torch.testing.assert_close(backend.score_features(features), scores, rtol=0, atol=0)
    torch.testing.assert_close(backend.score_tokens(tokens), scores, rtol=0, atol=0)
    return {
        "status": "passed", "dim": backend.dim, "token_dim": backend.token_dim,
        "features_shape": list(features.shape), "production_score_max_abs_error": float((scores - old_scores).abs().max()),
        "head_export_roundtrip_exact": True, "query_hashes_unchanged": True,
        "head_parameters_frozen": True, "branch_input_gradient_norms": branch_gradients,
        "nonzero_feature_change_max_logit_change": change, "encoder_release_parity_exact": True,
    }


def synthetic(module):
    old = module._token_module().TokenBackend.__new__(module._token_module().TokenBackend)
    old.device = torch.device("cpu")
    old.native = object()
    old.class_names = None
    old.metadata = {"source_files": [], "aggregation_origin": "added"}
    classes = ("Pneumonia", "Pneumothorax", "Pleural_Thickening")
    queries = torch.randn(2 * len(classes), module.BRANCH_DIM)
    if module.NORMALIZE:
        old.queries = F.normalize(queries, dim=-1)
    else:
        old.queries = queries * torch.linspace(0.2, 1.4, 2 * len(classes))[:, None]
        old.positive, old.negative = old.queries.chunk(2, 0)
        old.attention_queries = F.normalize(old.queries, dim=-1)

    def prepare_fixture(self, class_names):
        self.class_names = tuple(class_names)
    old.prepare_classes = types.MethodType(prepare_fixture, old)
    backend = module.DirectBackend.__new__(module.DirectBackend)
    backend.token_backend = old
    backend.device = old.device
    backend.class_names = None
    backend.head = None
    backend.prepare_classes(classes)
    tokens = torch.randn(2, old.tokens_per_image, module.BRANCH_DIM)
    tokens *= torch.linspace(0.1, 3.0, old.tokens_per_image)[None, :, None]
    return check(module, backend, tokens)


def real_cache(module, cache_dir):
    metadata = json.loads((cache_dir / "cache.json").read_text())
    dataset = json.loads(Path(metadata["dataset_path"]).read_text())
    samples_path = Path(dataset["samples_file"])
    if not samples_path.is_absolute():
        samples_path = Path(metadata["dataset_path"]).parent / samples_path
    samples = json.loads(samples_path.read_text())
    indices = [index for index, sample in enumerate(samples) if sample["split"] == "train"][:2]
    assert len(indices) == 2
    backend = module.DirectBackend(device="cpu")
    backend.prepare_classes(metadata["class_names"])
    tokens = torch.from_numpy(np.array(np.load(cache_dir / "tokens.npy", mmap_mode="r")[indices], copy=True))
    stored = torch.from_numpy(np.array(np.load(cache_dir / "aggregation_logits.npy", mmap_mode="r")[indices], copy=True))
    scores = backend.score_tokens(tokens)
    torch.testing.assert_close(scores, stored, rtol=2e-4, atol=2e-4)
    result = check(module, backend, tokens)
    result.update(train_indices=indices, cache=str(cache_dir), gpu_cached_score_max_abs_error=float((scores - stored).abs().max()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--real-cache", type=Path)
    args = parser.parse_args()
    if args.real_cache and not args.model:
        parser.error("--real-cache requires --model")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results = {}
    for name in (args.model,) if args.model else MODELS:
        module = load(BASELINES / name / "local_setup/direct_anchor_adapter.py", "direct_test_" + name.replace("-", "_"))
        results[name] = {"synthetic": synthetic(module)}
        if args.real_cache:
            results[name]["real_cache"] = real_cache(module, args.real_cache)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
