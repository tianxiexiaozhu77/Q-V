#!/usr/bin/env python3
"""Export frozen AlphaRAD features and recompute native-query controls.

Only support labels enter the fit. Every changed query is sent back through
the original FLaS attention/matching branch. Original-view projected FP32 tokens
may be read from the certified old cache; center images are freshly encoded.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time
import traceback

from cache_native_common import (MEDICAL, BRANCHES, parse_arguments, load_protocol,
    write_json, sha, identity_arrays, save_features, progress,
    assert_prediction_parity, unit)

ALPHA = MEDICAL / "AlphaRAD"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"


def write_images(path, rows):
    # Labels and boxes deliberately never enter the native image loader.
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        writer.writerows([[row["image"]] for row in rows])


def load_old_cache(protocol, test_all, evaluator):
    """Reuse only certified unchanged image tokens, with fresh input identities."""
    from vision_cache import VisionTokenCache, image_fingerprints

    root = Path(protocol["references"]["AlphaRAD"]["path"]).parent
    info_path = root / "vision_cache.json"
    if not info_path.is_file():
        return None, {"active": False, "reason": "No certified previous cache record"}
    info = json.loads(info_path.read_text())
    manifest_path = Path(info["path"]) / "manifest.json"
    if not info.get("certified") or sha(manifest_path) != info["manifest_sha256"]:
        raise ValueError("Previously certified vision cache changed")
    manifest = json.loads(manifest_path.read_text())
    paths = [row["image"] for row in test_all]
    identity = manifest["identity"]
    same = (manifest["status"] == "complete" and manifest["image_ids"] == paths
        and identity["checkpoint_sha256"] == CHECKPOINT_SHA
        and identity["use_all_image_tokens"] == evaluator.use_all_image_tokens
        and identity["num_heads"] == evaluator.model.num_radzero_heads
        and identity["input_file_fingerprints"] == image_fingerprints(paths))
    if not same:
        return None, {"active": False, "reason": "Cached image/model identities differ; fresh encoding required"}
    cache = VisionTokenCache(manifest_path.parent.parent, identity, paths, mode="read", prefetch=1)
    return cache, {"active": True, "path": str(manifest_path.parent),
        "manifest_sha256": sha(manifest_path), "mode": "read_only",
        "verification": "input identities, complete certificate, fresh first batch and every-image old prediction parity"}


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F

    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import disease_name_mapping
    from query_refinement import (QueryRefinementMatcher, original_queries_from_evaluator,
                                  fit_query_directions)
    from eval_query_refinement import original_helper_check
    from view_geometry import apply_view, aggregate_outputs

    started = time.time()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device is unavailable")
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Frozen published AlphaRAD checkpoint changed")
    protocol, support, test_all = load_protocol(args.protocol)
    test = test_all[:args.limit] if args.limit else test_all
    classes = protocol["class_names"]
    maps_enabled = protocol["dataset"] == "chestxdet10"
    reference = protocol["references"]["AlphaRAD"]
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Old AlphaRAD reference result changed")
    for name, rows in (("support", support), ("test", test)):
        write_images(args.output / f"{name}_inputs.csv", rows)
    options = {"batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": True, "shuffle": False, "drop_last": False}
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(CHECKPOINT)]},
        "data_test": {name: {"name": name, "data_type": "image_classification",
            "data_path": str(args.output / f"{name}_inputs.csv"),
            "n_class": len(classes), "normalize": None} for name in ("support", "test")},
        "dataloader": {"test": options}}
    write_json(args.output / "eval_config.json", config)
    progress(args, "AlphaRAD", "loading")
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored_keys = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    positive_raw, positive_prompts = original_queries_from_evaluator(evaluator, classes)
    negative_prompts = [f"There is no {disease_name_mapping(name)}." for name in classes]
    negative_raw = np.asarray(evaluator.encode_text(negative_prompts,
        text_max_length=evaluator.config["base"]["dqn_label_max_length"])["cls_emb_raw"], dtype=np.float32)
    if negative_raw.ndim == 1:
        negative_raw = negative_raw[None]
    matcher = QueryRefinementMatcher(evaluator.model, positive_raw[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)

    def global_feature(vision):
        # Native frozen patch projection, class independent. This global vector
        # is solely for feature-based comparator methods, never a substitute for
        # native attention when evaluating modified queries.
        if len(vision["image_patch_raw_features"]) != 1:
            raise ValueError("Expected the published single projected feature layer")
        return F.normalize(vision["image_patch_raw_features"][0].mean(1), dim=-1)

    support_parts = {key: [] for key in ("features", "local", "logits")}
    observed = []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support"]:
            vision = matcher.encode_vision(batch["images"])
            result = matcher.match_prepared(vision, positive_raw, return_features=True)
            support_parts["features"].append(global_feature(vision).cpu().numpy())
            support_parts["local"].append(result["class_local"].cpu().numpy())
            support_parts["logits"].append(result["logits"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [row["image"] for row in support]:
        raise ValueError("Support order changed")
    support_arrays = {key: np.concatenate(value) for key, value in support_parts.items()}
    support_labels = np.asarray([row["labels"] for row in support], dtype=np.uint8)
    fit = fit_query_directions(support_arrays["local"], support_labels, positive_raw, etas=(0., .3))
    query = torch.as_tensor(fit["queries"][1], device=matcher.device)
    positive_query = F.normalize(F.normalize(torch.as_tensor(positive_raw, device=matcher.device), dim=-1)
        + .3 * F.normalize(torch.as_tensor(fit["positive_means"], device=matcher.device), dim=-1), dim=-1)
    save_features(args.output / "support.npz", support, classes, support_arrays, support=True)
    np.savez_compressed(args.output / "queries.npz", positive=unit(positive_raw), negative=unit(negative_raw),
        positive_raw=positive_raw, negative_raw=negative_raw, query=query.cpu().numpy(),
        positive_query=positive_query.cpu().numpy(), positive_prompts=np.asarray(positive_prompts),
        negative_prompts=np.asarray(negative_prompts), class_names=np.asarray(classes))
    write_json(args.output / "query_fit.json", {"eta": .3, "support_count": len(support),
        "positive_counts": fit["positive_counts"].tolist(), "negative_counts": fit["negative_counts"].tolist(),
        "positive_prototype": "unit positive original-query context mean; eta residual; unit query as old AlphaRAD protocol",
        "negative_text_usage": "Comparator feature-space reference only; native AlphaRAD FLaS original score does not subtract it",
        "test_labels_used": False})

    cache, cache_record = load_old_cache(protocol, test_all, evaluator)
    arrays = {key: [] for key in BRANCHES}
    maps = {key: [] for key in BRANCHES} if maps_enabled else {}
    features_out = {key: [] for key in ("features", "local", "logits")}
    parity, offset = {}, 0
    try:
        with torch.inference_mode():
            for batch in evaluator.data_loader_dict["test"]:
                stop = offset + len(batch["images"])
                if batch["image_paths"] != [row["image"] for row in test[offset:stop]]:
                    raise ValueError("Test image order changed")
                if cache is not None:
                    tokens = torch.as_tensor(np.array(cache.get(offset, stop), copy=True),
                                             dtype=torch.float32, device=matcher.device)
                    vision = {"image_cls_raw_features": [tokens[:, 0]], "image_patch_raw_features": [tokens[:, 1:]]}
                    if offset == 0:
                        fresh = matcher.encode_vision(batch["images"])
                        differences = {key: float((vision[key][0] - fresh[key][0]).abs().max().item())
                                       for key in vision}
                        try:
                            for key in vision:
                                torch.testing.assert_close(vision[key][0], fresh[key][0], atol=2e-5, rtol=1e-5)
                        except AssertionError:
                            # In particular, smoke prefixes can change encoder
                            # GEMM batch shape. Do not weaken cache certification:
                            # use current freshly encoded features for this entire
                            # run and retain the final every-image score parity.
                            cache.close()
                            cache = None
                            vision = fresh
                            cache_record.update(active=False,
                                first_batch_fresh_token_check=False,
                                first_batch_max_abs_error=differences,
                                reason="Strict first-batch token check failed; switched to fresh encoding for every test image")
                        else:
                            cache_record["first_batch_fresh_token_check"] = True
                            cache_record["first_batch_max_abs_error"] = differences
                else:
                    vision = matcher.encode_vision(batch["images"])
                original = matcher.match_prepared(vision, positive_raw, return_maps=maps_enabled, return_features=True)
                refined = matcher.match_prepared(vision, query, return_maps=maps_enabled)
                pos_result = matcher.match_prepared(vision, positive_query, return_maps=maps_enabled)
                center = matcher.encode_vision(apply_view(batch["images"], "center80"))
                center_original = matcher.match_prepared(center, positive_raw, return_maps=maps_enabled)
                center_refined = matcher.match_prepared(center, query, return_maps=maps_enabled)
                outputs = {"original": original, "query": refined, "positive_query": pos_result,
                    "view": aggregate_outputs({"identity": original, "center80": center_original}, ["identity", "center80"]),
                    "query_view": aggregate_outputs({"identity": refined, "center80": center_refined}, ["identity", "center80"])}
                if offset == 0:
                    parity["native_author_helpers"] = original_helper_check(evaluator, matcher, batch, classes)
                for name, output in outputs.items():
                    arrays[name].append(torch.sigmoid(output["logits"]).cpu().numpy())
                    if maps_enabled:
                        maps[name].append(output["grounding_map"].cpu().numpy())
                features_out["features"].append(global_feature(vision).cpu().numpy())
                features_out["local"].append(original["class_local"].cpu().numpy())
                features_out["logits"].append(original["logits"].cpu().numpy())
                if offset == 0 or (offset // args.batch_size) % 25 == 0 or stop == len(test):
                    progress(args, "AlphaRAD", "test", stop, len(test))
                offset = stop
    finally:
        if cache is not None:
            cache.close()
    arrays = {key: np.concatenate(value).astype(np.float32) for key, value in arrays.items()}
    feature_arrays = {key: np.concatenate(value) for key, value in features_out.items()}
    save_features(args.output / "test.npz", test, classes, feature_arrays)
    old_root = Path(reference["path"]).parent
    old_paths = json.loads((old_root / "image_paths.json").read_text())
    lookup = {str(Path(path).resolve()): i for i, path in enumerate(old_paths)}
    indices = [lookup[str(Path(row["image"]).resolve())] for row in test]
    for name, old_name in (("original", "original"), ("query", "uniform_eta03"),
                          ("view", "tta_center_original"), ("query_view", "tta_center_refined")):
        old = np.load(old_root / "candidates" / old_name / "probabilities.npy", allow_pickle=False)
        parity[name] = assert_prediction_parity(arrays[name], old[indices], f"old AlphaRAD {name}")
    np.savez_compressed(args.output / "native_predictions.npz", **identity_arrays(test, classes), **arrays)
    if maps_enabled:
        np.savez_compressed(args.output / "native_maps.npz", **identity_arrays(test, classes),
            **{key: np.concatenate(value).astype(np.float32) for key, value in maps.items()})
    write_json(args.output / "native_parity.json", parity)
    write_json(args.output / "vision_cache.json", cache_record)
    if any(parameter.requires_grad for parameter in evaluator.model.parameters()):
        raise ValueError("The frozen AlphaRAD backbone contains trainable parameters")
    result = {"status": "smoke_complete" if args.limit else "complete", "model": "AlphaRAD",
        "dataset": protocol["dataset"], "support_count": len(support), "test_count": len(test),
        "full_test_count": len(test_all), "class_names": classes,
        "feature_dimension": feature_arrays["features"].shape[-1],
        "feature_definition": "unit mean of native projected patches; local is original-query FLaS head context concatenated and globally unit-normalized",
        "logits_definition": "unchanged native FLaS binary presence logit; not positive-minus-negative cosine",
        "score_domain": "native sigmoid probability; two views average native logits before sigmoid",
        "query_norm_policy": "old AlphaRAD fit: original raw query at eta=0; globally unit query at eta=.3",
        "native_map_definition": "raw scaled native FLaS map including CLS; center map inverse-warp and valid-coverage average" if maps_enabled else None,
        "test_labels_used": False, "test_boxes_used": False,
        "protocol_sha256": sha(args.protocol), "checkpoint_sha256": CHECKPOINT_SHA,
        "elapsed_seconds": time.time() - started, "device": torch.cuda.get_device_name(0),
        "physical_gpu": args.gpu, "native_parity": parity, "vision_cache": cache_record,
        "ignored_checkpoint_keys": ignored_keys,
        "source_sha256": {str(path): sha(path) for path in [Path(__file__),
            Path(__file__).with_name("cache_native_common.py"), ALPHA / "local_setup/query_refinement.py",
            ALPHA / "local_setup/eval_query_refinement.py", PHASE6 / "view_geometry.py"]}}
    write_json(args.output / "result.json", result)
    write_json(args.output / "status.json", {"status": result["status"], "model": "AlphaRAD",
        "dataset": protocol["dataset"], "done": len(test), "total": len(test_all)})
    print(json.dumps({key: result[key] for key in ("status", "model", "dataset", "test_count", "elapsed_seconds")}), flush=True)


def main():
    args = parse_arguments(__doc__)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
