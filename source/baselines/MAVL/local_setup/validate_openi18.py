"""Independently validate fixed MAVL OpenI18 queries and the released shared head.

Only two TRAIN images are used.  This script reserves one authorized physical
GPU via the shared lease helper and never selects wording or model parameters.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/baseline_token_classification"))
from common import DATASETS, load_backend, read_dataset, release_gpu, residual_mlp, save, setup_gpu, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock, gpu = setup_gpu(args.gpu)
    started = time.time()
    report = {"status": "running", "model": "MAVL", "device": "cuda", "gpu": gpu, "checks": {},
              "backend_variant": "openi18_query_extension",
              "validation_source_sha256": sha256(__file__),
              "uses_only_train_images": True,
              "numerical_tolerance": {"atol": 5e-5, "rtol": 1e-5}}
    try:
        import torch
        import torchvision.models as tvmodels

        def close(name, actual, expected, atol=5e-5, rtol=1e-5):
            if actual.shape != expected.shape:
                raise AssertionError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
            if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                raise AssertionError(name + ": nonfinite values")
            error = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
            report["checks"][name] = {"shape": list(actual.shape), "max_absolute_error": error,
                                      "atol": atol, "rtol": rtol}
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)

        meta, rows = read_dataset(DATASETS["openi"])
        classes = meta["class_names"]
        if len(classes) != 18 or classes[-2:] != ["Calcified Granuloma", "Granuloma"]:
            raise AssertionError("Expected the frozen full OpenI18 label order")
        selected = [row for row in rows if row["split"] == "train"][:2]
        if len(selected) != 2:
            raise AssertionError("Need two TRAIN images")
        paths = [row["image"] for row in selected]
        report["selected_train_images"] = [{"id": row["id"], "split": row["split"], "image": row["image"],
                                            "image_sha256": sha256(row["image"])} for row in selected]
        report["dataset"] = str(DATASETS["openi"])
        report["dataset_sha256"] = sha256(DATASETS["openi"])

        backend = load_backend("MAVL", "cuda", variant="openi18_query_extension")
        backend.prepare_classes(classes)
        native = backend.native
        extension = sys.modules["mavl_openi18_native_extension"]
        base = extension._base
        resource = extension.load_resource()
        disease_book = json.loads((extension.AUTHOR / "observation explanation.json").read_text())
        concept_book = json.loads((extension.ROOT / "Pretrain/concept_gen/gpt4_mimic.json").read_text())
        original_classes = [key for key in disease_book if key != "covid19"]
        if len(original_classes) != 75:
            raise AssertionError("Original author disease book must have 75 feature classes")
        for key in resource["query_order"]:
            disease_book[key] = resource["queries"][key]["description"]
            concept_book[key] = resource["queries"][key]["concepts"]
        all_classes = original_classes + resource["query_order"]
        if len(all_classes) != 77 or any(len(concept_book[key]) != 8 for key in all_classes):
            raise AssertionError("Direct author construction needs 77 descriptions and 77 x 8 concepts")

        bert = str(extension.ROOT / "local_assets/Bio_ClinicalBERT")
        tokenizer = base.BertTokenizer.from_pretrained(bert, local_files_only=True)

        def tokenize(texts):
            return tokenizer(texts, padding="max_length", truncation=True,
                             max_length=64, return_tensors="pt").to(backend.device)

        config = dict(d_model=256, base_model="resnet50", decoder="cross", num_queries=75,
                      dropout=.1, attribute_set_size=2, N=4, H=4, self_attention=True,
                      text_encoder=bert, pretrained=False)
        author_class = base.MAVL
        original_get_basemodel = author_class._get_basemodel

        def chosen_resnet(self, model_name, pretrained=False, layers=None):
            if model_name != "resnet50":
                raise AssertionError("Unexpected visual architecture")
            network = tvmodels.resnet50(weights=None)
            return torch.nn.Sequential(*list(network.children())[:-3]), int(network.fc.in_features / 2)

        author_class._get_basemodel = chosen_resnet
        torch.manual_seed(42)
        try:
            # The unmodified author constructor performs its own book encoding
            # and description-first nine-slot assembly for all 77 diseases.
            author = author_class(
                config,
                tokenize(["It is located at " + key for key in base.ANATOMIES]),
                tokenize([disease_book[key] for key in all_classes]),
                tokenize([text for key in all_classes for text in concept_book[key]]),
            )
        finally:
            author_class._get_basemodel = original_get_basemodel
        checkpoint = extension.ROOT / "local_assets/checkpoint_full_46.pth"
        checkpoint_state = torch.load(checkpoint, map_location="cpu")["model"]
        checkpoint_state = {key.removeprefix("module."): value for key, value in checkpoint_state.items()
                            if "temp" not in key}
        author.load_state_dict(checkpoint_state, strict=True)
        author = author.to(backend.device).eval().requires_grad_(False)
        author_state = author.state_dict()
        overlay_state = backend.model.state_dict()
        if author_state.keys() != overlay_state.keys():
            raise AssertionError("Author and overlay checkpoint keys differ")
        if any(not torch.equal(author_state[key], overlay_state[key]) for key in author_state):
            raise AssertionError("Author and overlay checkpoint tensor values differ")
        report["strict_author_checkpoint_load"] = True
        report["checkpoint_state_values_bitwise_equal"] = True
        report["checks"]["author_reference_strict_checkpoint_load_and_state_equality"] = True
        report["checkpoint_sha256"] = {str(checkpoint): sha256(checkpoint)}

        # Independently encode the 693 texts in a single description-first list.
        # This checks the author's assembly as well as the overlay's append path.
        flat_texts = [text for key in all_classes for text in [disease_book[key], *concept_book[key]]]
        expected_slots = ["description", "border", "fluid", "location", "opacity", "other", "patterns", "shape", "texture"]
        if len(flat_texts) != 693:
            raise AssertionError("Expected exactly 693 slot texts")
        for key in ("calcified granuloma",):
            prefixes = [text.split(":", 1)[0] for text in concept_book[key]]
            if prefixes != expected_slots[1:]:
                raise AssertionError("Author Calcified Granuloma aspect order changed")
        if flat_texts[684:693] != [disease_book["granuloma"]] * 9:
            raise AssertionError("Granuloma must use the one fixed statement in all nine slots")
        encoder = author._get_bert_basemodel(bert, freeze_layers=None)
        encoder = encoder.to(backend.device).eval().requires_grad_(False)
        with torch.no_grad():
            tokens = tokenize(flat_texts)
            one_pass_raw = encoder(input_ids=tokens["input_ids"],
                                   attention_mask=tokens["attention_mask"]).last_hidden_state[:, 0, :]
            one_pass_projected = author.disease_embedding_layer(one_pass_raw)
            close("author_raw_queries_vs_one_pass_693", author.concept_book, one_pass_raw)
            close("overlay_raw_queries_vs_one_pass_693", backend.model.concept_book, one_pass_raw)
            close("author_projected_queries_vs_one_pass_693",
                  author.disease_embedding_layer(author.concept_book), one_pass_projected)
            close("overlay_projected_queries_vs_one_pass_693", backend.queries, one_pass_projected)
            close("author_vs_overlay_raw_queries", author.concept_book, backend.model.concept_book)
            close("author_vs_overlay_disease_descriptions", author.disease_book, backend.model.disease_book)
            for offset, key in ((675, "calcified granuloma"), (684, "granuloma")):
                close(key.replace(" ", "_") + "_nine_slot_order",
                      backend.model.concept_book[offset:offset + 9], one_pass_raw[offset:offset + 9])
            pixels = backend._pixels(paths)
            author_logits = author(pixels, return_intermediate=False)
            overlay_logits = backend.model(pixels, return_intermediate=False)
            if tuple(author_logits.shape) != (2, 77, 2):
                raise AssertionError("Expected complete author output shape [2,77,2]")
            if backend.indices[-2:] != [75, 76] or len(set(backend.indices)) != 18:
                raise AssertionError("OpenI18 query extraction indices changed")
            close("author77_full_logits", overlay_logits, author_logits)
            expected_margin = (author_logits[..., 1] - author_logits[..., 0])[:, backend.indices]
            close("author77_openi18_native_scores", backend.native_scores(paths), expected_margin)
            close("author77_openi18_native_probabilities", backend.native_predictions(paths),
                  author_logits.softmax(dim=-1)[:, backend.indices, 1])
            if native.classes != all_classes or native.model.n_disease != 77:
                raise AssertionError("native_predictions unexpectedly reset the extended context")
            visual_tokens = backend.encode_images(paths)
            token_scores = backend.score_tokens(visual_tokens)
            close("author77_openi18_cached_token_scores", token_scores, expected_margin)
            close("native_margin_sigmoid_equals_probability", backend.native_scores(paths).sigmoid(),
                  backend.native_predictions(paths), atol=2e-6)
            torch.manual_seed(42)
            adapter = residual_mlp(backend.dim, 64).to(backend.device).eval()
            close("zero_adapter_identity_tokens", adapter(visual_tokens), visual_tokens, atol=0, rtol=0)
            close("zero_adapter_matches_extended_native_scores", backend.score_tokens(visual_tokens, adapter),
                  token_scores, atol=0, rtol=0)
        if any(parameter.requires_grad for parameter in backend.model.parameters()):
            raise AssertionError("An original model parameter became trainable")
        report["checks"]["overlay_original75_prefix_bitwise_equal"] = backend.metadata["original_queries_preserved"]
        report["checks"]["overlay_original675_concept_prefix_bitwise_equal"] = backend.metadata["extension_integrity"]["original_675_concept_queries_bitwise_equal"]
        report["checks"]["nine_slot_order_verified"] = True
        report["checks"]["all_native_calls_preserve_fixed77_context"] = True
        report["checks"]["all_model_parameters_frozen"] = True
        source_files = [*backend.metadata["source_files"], str(Path(__file__).resolve()),
                        str(ROOT / "experiments/baseline_token_classification/common.py")]
        report["source_sha256"] = {path: sha256(path) for path in source_files}
        report.update(status="passed", class_names=classes, all_query_classes=all_classes,
                      class_query_indices=backend.indices,
                      evaluation_scope="Exactly two fixed TRAIN images; no TEST images or performance metrics",
                      query_context_count=693, original_query_count=75,
                      nine_slot_order=expected_slots,
                      nine_slot_order_verified=True,
                      original_queries_preserved=backend.metadata["original_queries_preserved"],
                      original_extension_integrity=backend.metadata["extension_integrity"],
                      untruncated_token_lengths=backend.metadata["untruncated_token_lengths"],
                      resource_sha256=backend.metadata["resource_sha256"],
                      backend_metadata=backend.metadata,
                      seconds=time.time() - started)
        save(args.output, report)
        print(json.dumps({"status": "passed", "output": str(args.output),
                          "max_abs_error": max(value["max_absolute_error"] for value in report["checks"].values() if isinstance(value, dict)),
                          "checks": len(report["checks"])}, ensure_ascii=False), flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc(), seconds=time.time() - started)
        save(args.output, report)
        raise
    finally:
        release_gpu(lock)


if __name__ == "__main__":
    main()
