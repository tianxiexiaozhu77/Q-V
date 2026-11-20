"""Validate the fixed MedKLIP OpenI18 extension against the author model.

Uses exactly two TRAIN images, no TEST metrics, and the common physical GPU
4--7 lease. This validates implementation parity, not disease accuracy.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[2]
COMMON_DIR = MEDICAL / "experiments/baseline_token_classification"
sys.path.insert(0, str(COMMON_DIR))
from common import DATASETS, read_dataset, release_gpu, residual_mlp, save, setup_gpu, sha256


def validate(gpu):
    import torch

    started = time.time()
    path = HERE / "openi18_token_mlp_adapter.py"
    spec = importlib.util.spec_from_file_location("medklip_openi18_native_reference_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ext = module.EXTENSION
    meta, rows = read_dataset(DATASETS["openi"])
    classes = meta["class_names"]
    if len(classes) != 18 or classes[-2:] != ["Calcified Granuloma", "Granuloma"]:
        raise ValueError("Unexpected full OpenI18 class order")
    selected = [row for row in rows if row["split"] == "train"][:2]
    if len(selected) != 2:
        raise ValueError("Two TRAIN images are required")
    paths = [row["image"] for row in selected]
    backend = module.TokenBackend(device="cuda").prepare_classes(classes)
    overlay = backend.native
    checks = {}

    def close(name, actual, expected, atol=1e-4, rtol=1e-5):
        if tuple(actual.shape) != tuple(expected.shape):
            raise AssertionError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{name}: nonfinite values")
        error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        checks[name] = {"max_absolute_error": error, "atol": atol, "rtol": rtol,
                        "shape": list(actual.shape)}

    with torch.no_grad():
        # This reference is the unmodified author class, constructed once with
        # the entire 77-description list rather than concatenating embeddings.
        descriptions = list(overlay.book.values()) + list(overlay.extra_book.values())
        if len(descriptions) != 77:
            raise AssertionError("Reference does not contain exactly 77 descriptions")
        tokens = overlay.tokenizer(descriptions, padding="max_length", truncation=True,
                                   max_length=64, return_tensors="pt").to("cuda")
        config = dict(d_model=256, res_base_model="resnet50", H=4, N=4, dropout=.1,
                      attribute_set_size=2, num_queries=77,
                      text_encoder=str(ext.ROOT / "local_assets/Bio_ClinicalBERT"))
        torch.manual_seed(42)
        reference = ext.NATIVE_BASE.MedKLIP(config, tokens)
        state = torch.load(overlay.checkpoint, map_location="cpu")["model"]
        loaded = reference.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise AssertionError("Reference checkpoint was not strictly loaded")
        del state
        reference = reference.to("cuda").eval().requires_grad_(False)
        if ext._state_digest(reference) != ext._state_digest(overlay.model):
            raise AssertionError("Reference and overlay checkpoint tensors differ")
        checks["author_reference_strict_checkpoint_load_and_state_equality"] = True
        if not torch.equal(overlay.model.disease_book[:75], overlay.base_queries):
            raise AssertionError("Original 75 query embeddings were modified")
        checks["overlay_original75_prefix_bitwise_equal"] = True
        # BERT batch sizes differ (75+2 versus 77); tolerate float32 rounding.
        close("author77_original75_raw_embeddings", overlay.model.disease_book[:75],
              reference.disease_book[:75], atol=3e-5)
        close("author77_new2_raw_embeddings", overlay.model.disease_book[75:],
              reference.disease_book[75:], atol=3e-5)
        close("author77_projected_queries", backend.queries,
              reference.disease_embedding_layer(reference.disease_book), atol=3e-5)

        pixels = backend._pixels(paths)
        reference_logits = reference(pixels)
        overlay_logits = overlay.model(pixels)
        if tuple(reference_logits.shape) != (2, 77, 2):
            raise AssertionError("Author reference output must be [2,77,2]")
        close("author77_full_logits", overlay_logits, reference_logits)
        indices = overlay._indices(classes)
        if indices[-2:] != [75, 76] or len(set(indices)) != 18:
            raise AssertionError("Incorrect OpenI18 query extraction")
        reference_scores = (reference_logits[..., 1] - reference_logits[..., 0])[:, indices]
        close("author77_openi18_native_scores", backend.native_scores(paths), reference_scores)
        reference_probabilities = reference_logits.softmax(-1)[:, indices, 1]
        close("author77_openi18_native_probabilities", backend.native_predictions(paths),
              reference_probabilities, atol=1e-5)
        image_tokens = backend.encode_images(paths)
        close("author77_openi18_cached_token_scores", backend.score_tokens(image_tokens), reference_scores)
        close("native_margin_sigmoid_equals_probability", backend.native_scores(paths).sigmoid(),
              backend.native_predictions(paths), atol=2e-6)
        zero_adapter = residual_mlp(backend.dim).to("cuda").eval().requires_grad_(False)
        close("zero_adapter_identity_tokens", zero_adapter(image_tokens), image_tokens, atol=0, rtol=0)
        close("zero_adapter_matches_extended_native_scores", backend.score_tokens(image_tokens, zero_adapter),
              backend.native_scores_from_tokens(image_tokens), atol=0, rtol=0)
        original_query_pointer = overlay.model.disease_book
        overlay._activate_queries(False)
        if overlay.model.disease_book is not original_query_pointer:
            raise AssertionError("Native query activation reset the OpenI context")
        try:
            overlay._activate_queries(True)
        except ValueError:
            pass
        else:
            raise AssertionError("COVID activation was allowed in the fixed OpenI context")
        overlay._check_context()
        if any(p.requires_grad for p in overlay.model.parameters()):
            raise AssertionError("A frozen model parameter unexpectedly became trainable")
        checks["all_native_calls_preserve_fixed77_context"] = True
        checks["all_model_parameters_frozen"] = True

    source_files = list(backend.metadata["source_files"]) + [str(Path(__file__).resolve()),
                                                            str(COMMON_DIR / "common.py")]
    return {
        "status": "passed", "model": "MedKLIP", "device": "cuda", "gpu": gpu,
        "backend_variant": "openi18_query_extension", "seconds": time.time() - started,
        "evaluation_scope": "Exactly two fixed TRAIN images; no TEST images or performance metrics",
        "selected_train_images": [{"id": row["id"], "image": row["image"],
                                    "split": row["split"], "image_sha256": sha256(row["image"])}
                                   for row in selected],
        "dataset": str(DATASETS["openi"]), "dataset_sha256": sha256(DATASETS["openi"]),
        "class_names": classes, "class_query_indices": indices,
        "checks": checks, "backend_metadata": backend.metadata,
        "source_sha256": {path: sha256(path) for path in source_files},
        "checkpoint_sha256": {path: sha256(path) for path in backend.metadata["checkpoint_files"]},
        "notes": [
            "Original query prefix equality is checked against the 75-query embeddings before extension.",
            "The author's single 77-description BERT batch is compared numerically, since batch-size-dependent float32 rounding can occur.",
            "This test does not assert that 75-query and 77-query class predictions are unchanged: the native decoder has query self-attention.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = None
    try:
        lock, gpu = setup_gpu(args.gpu)
        result = validate(gpu)
        save(args.output, result)
        print("MedKLIP OpenI18 native77 validation passed:", args.output, flush=True)
    except Exception as error:
        save(args.output, {"status": "failed", "model": "MedKLIP", "physical_gpu": args.gpu,
                           "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if lock is not None:
            release_gpu(lock)


if __name__ == "__main__":
    main()
