#!/usr/bin/env python3
"""Export certified RadZero features and evaluate a positive-query control.

The four historical native branches are authenticated and kept unchanged.
The positive-only query is genuinely passed through the original cross-attention
on certified visual tokens. A fresh first-image encoder check and complete
original/query score parity certify that reuse. No test labels enter fitting.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time
import traceback

from cache_native_common import (MEDICAL, parse_arguments, load_protocol, load_module,
    write_json, sha, identity_arrays, save_features, progress, unit,
    assert_prediction_parity)

RADZERO = MEDICAL / "RadZero_i2cl"
SETUP = RADZERO / "local_setup"
HISTORY = MEDICAL / "experiments/query_search_20261021_152010"
DATASETS = {"chestxdet10": "ChestXDet10", "rsna_pneumonia": "RSNA"}
HISTORICAL_BRANCHES = {"original": "original", "query": "uniform_eta03",
                       "view": "tta_center_original", "query_view": "tta_center_refined"}


def checked_json(path, digest=None):
    path = Path(path)
    if digest is not None and sha(path) != digest:
        raise ValueError(f"Frozen JSON source changed: {path}")
    return json.loads(path.read_text())


def checked_array(path, record):
    import numpy as np
    path = Path(path)
    if sha(path) != record["sha256"]:
        raise ValueError(f"Frozen array checksum changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if "shape" in record and list(array.shape) != record["shape"]:
        raise ValueError(f"Frozen array shape changed: {path}")
    if "dtype" in record and str(array.dtype) != record["dtype"]:
        raise ValueError(f"Frozen array dtype changed: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite frozen array: {path}")
    return array


def inspect_inputs(protocol_path):
    """Authenticate feature/source identities on CPU, without parsing test labels."""
    import numpy as np
    protocol, support, test = load_protocol(protocol_path)
    dataset = DATASETS.get(protocol["dataset"])
    if dataset is None:
        raise ValueError("The locked first-round RadZero protocol supports ChestXDet10/RSNA")
    ref = protocol["references"]["RadZero"]
    reference_path = Path(ref["path"])
    reference = checked_json(reference_path, ref["sha256"])
    config = reference["config"]
    if reference["status"] != "complete" or reference["dataset"] != dataset:
        raise ValueError("Historical RadZero result is incomplete or belongs to another dataset")
    if config["class_names"] != protocol["class_names"] or reference["rows"] != len(test):
        raise ValueError("Historical RadZero class order or test count changed")
    if reference["support_count"] != len(support):
        raise ValueError("Historical RadZero support count differs")
    cache_path = Path(config["cache_manifest"]).parent
    manifest = checked_json(config["cache_manifest"], config["cache_manifest_sha256"])
    if manifest["status"] != "complete" or manifest["class_names"] != protocol["class_names"]:
        raise ValueError("Invalid frozen compact-feature manifest")
    for key in ("checkpoint", "support_identity", "test_identity", "split_sha256"):
        if manifest[key] != config[key]:
            raise ValueError(f"Historical configuration/cache mismatch: {key}")
    for path, digest in {**manifest["split_sha256"], **manifest["source_sha256"]}.items():
        if sha(path) != digest:
            raise ValueError(f"Certified source/split has changed: {path}")
    model_root = RADZERO / "local_assets/model"
    for file, key in (("config.json", "model_config_sha256"),
                      ("preprocessor_config.json", "preprocessor_sha256")):
        if sha(model_root / file) != manifest[key]:
            raise ValueError(f"Cached model configuration changed: {file}")

    metadata, arrays = {}, {}
    provenance = {str(reference_path): ref["sha256"],
                  str(cache_path / "manifest.json"): config["cache_manifest_sha256"]}
    for split, canonical in (("support", support), ("test", test)):
        meta_path = cache_path / f"{split}_metadata.json"
        meta_hash = manifest["auxiliary_files"][meta_path.name]["sha256"]
        metadata[split] = checked_json(meta_path, meta_hash)
        if "labels" in metadata[split]:
            raise ValueError("Compact metadata must not contain test labels")
        native_csv = Path(manifest["split_dir"]) / dataset / f"{split}.csv"
        # Select identity columns only; labels_json from test.csv is never used.
        with native_csv.open(newline="") as handle:
            rows = [{k: r[k] for k in ("image_path", "image_id", "patient_id", "group", "byte_sha256")}
                    for r in csv.DictReader(handle)]
        id_key = "patient_id" if dataset == "RSNA" else "image_id"
        if ([Path(row["image"]).stem for row in canonical] != [r[id_key] for r in rows]
                or [r["image_path"] for r in rows] != metadata[split]["image_paths"]
                or [r["image_id"] for r in rows] != metadata[split]["ids"]
                or [r["group"] for r in rows] != metadata[split]["groups"]):
            raise ValueError(f"Canonical/native/{split} cache identities or order disagree")
        if split == "support":
            if [r["group"] for r in rows] != [r["group"] for r in canonical]:
                raise ValueError("Support grouping differs between native/canonical data")
            root = Path(metadata[split]["data_root"])
            for row in rows:
                if sha(root / row["image_path"]) != row["byte_sha256"]:
                    raise ValueError("Native support image content changed")
        provenance[str(meta_path)] = meta_hash
        arrays[split] = {}
        for name in ("global_aligned", "class_local", "base_logits"):
            record = manifest["splits"][split]["files"][name]
            path = cache_path / record["file"]
            if path.parent != cache_path:
                raise ValueError("Feature cache record escapes its root")
            arrays[split][name] = checked_array(path, record)
            provenance[str(path)] = record["sha256"]
    for key in ("image_paths", "ids", "groups"):
        if set(metadata["support"][key]) & set(metadata["test"][key]):
            raise ValueError(f"Native support/test overlap: {key}")
    for name in ("support_labels", "text_features"):
        path = cache_path / f"{name}.npy"
        record = manifest["auxiliary_files"][path.name]
        arrays[name] = checked_array(path, record)
        provenance[str(path)] = record["sha256"]
    np.testing.assert_array_equal(arrays["support_labels"], np.asarray([r["labels"] for r in support]))
    if config["prompts"] != metadata["support"]["prompts"] or config["prompts"] != metadata["test"]["prompts"]:
        raise ValueError("Native prompt identity changed")

    branch_arrays = {}
    for name, historical_name in HISTORICAL_BRANCHES.items():
        record = reference["candidates"][historical_name]
        path = Path(record["directory"]) / "logits.npy"
        branch_arrays[name] = checked_array(path, {"sha256": record["logits_sha256"],
            "shape": [len(test), len(protocol["class_names"])], "dtype": "float32"})
        provenance[str(path)] = record["logits_sha256"]
    np.testing.assert_array_equal(branch_arrays["original"], arrays["test"]["base_logits"])

    token_info = config["vision_cache"]
    tokens = checked_json(token_info["manifest"], token_info["manifest_sha256"])
    if tokens["status"] != "complete" or tokens["cached_rows"] != len(test):
        raise ValueError("Visual-token cache is not certified complete")
    for key in ("checkpoint", "support_identity", "test_identity", "split_sha256",
                "model_config_sha256", "preprocessor_sha256"):
        if tokens["identity"][key] != manifest[key]:
            raise ValueError(f"Visual-token/compact-feature identity differs: {key}")
    if tokens["identity"]["model"] != "radzero" or tokens["identity"]["dataset"] != dataset:
        raise ValueError("Wrong visual-token model/dataset")
    if tokens["identity"]["stage"] != "aligned_raw_before_loss_layernorm_and_token_selection":
        raise ValueError("Wrong visual-token extraction stage")
    native_root = Path(metadata["test"]["data_root"])
    if len(tokens["image_ids"]) != len(test):
        raise ValueError("Visual cache identity list length differs")
    for path, recorded in zip(metadata["test"]["image_paths"], tokens["image_ids"]):
        actual = (native_root / path).resolve()
        stat = actual.stat()
        if json.loads(recorded) != [str(actual), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]:
            raise ValueError(f"Native source image identity/stat changed: {actual}")
    provenance[str(Path(token_info["manifest"]))] = token_info["manifest_sha256"]
    return dict(protocol=protocol, support=support, test=test, dataset=dataset,
                reference=reference, config=config, manifest=manifest, metadata=metadata,
                arrays=arrays, branch_arrays=branch_arrays, tokens=tokens,
                token_info=token_info, provenance=provenance)


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device is unavailable")
    progress(args, "RadZero", "authenticating_caches")
    inputs = inspect_inputs(args.protocol)
    protocol, support, test_all = inputs["protocol"], inputs["support"], inputs["test"]
    test = test_all[:args.limit] if args.limit else test_all
    classes, arrays = protocol["class_names"], inputs["arrays"]
    for split, rows in (("support", support), ("test", test)):
        source = arrays[split]
        save_features(args.output / f"{split}.npz", rows, classes, {
            "features": unit(source["global_aligned"][:len(rows), 1]),
            "local": source["class_local"][:len(rows)],
            "logits": source["base_logits"][:len(rows)]}, support=split == "support")

    sys.path.insert(0, str(SETUP))
    from infer_batch import load_model, CachedInference, TestImages, checkpoint_identity
    from overnight_query_refinement import QueryRefinementMatcher, fit_query_directions
    from extract_frozen_features import pool_original_matching
    if checkpoint_identity() != inputs["manifest"]["checkpoint"]:
        raise ValueError("Released checkpoint differs from certified features")
    progress(args, "RadZero", "loading_model")
    model, processor, tokenizer = load_model(torch.device("cuda"))
    model.eval().requires_grad_(False)
    positive_prompts = inputs["config"]["prompts"]
    if any(not prompt.startswith("There is ") for prompt in positive_prompts):
        raise ValueError("Explicit negative prompts require verified 'There is' native prompts")
    negative_prompts = [prompt.replace("There is ", "There is no ", 1) for prompt in positive_prompts]
    with torch.inference_mode():
        positive_encoder = CachedInference(model, tokenizer, positive_prompts)
        negative_encoder = CachedInference(model, tokenizer, negative_prompts)
        positive_raw = positive_encoder.features.detach()
        negative_raw = negative_encoder.features.detach()
        parity = {"positive_text": assert_prediction_parity(positive_raw.cpu().numpy(), arrays["text_features"], "native positive text")}
        fit = fit_query_directions(arrays["support"]["class_local"], arrays["support_labels"],
                                   arrays["text_features"], (0.0, 0.3))
        query = torch.as_tensor(fit["queries"][1], device="cuda")
        positive_mean = torch.as_tensor(fit["positive_means"], device="cuda")
        positive_only = F.normalize(F.normalize(positive_raw, dim=-1)
                                    + 0.3 * F.normalize(positive_mean, dim=-1), dim=-1)
        matcher = QueryRefinementMatcher(model, np.stack((fit["queries"][0], fit["queries"][1])), (0.0, 0.3))
    np.savez_compressed(args.output / "queries.npz", positive=unit(positive_raw.cpu().numpy()),
        negative=unit(negative_raw.cpu().numpy()), positive_raw=positive_raw.cpu().numpy(),
        negative_raw=negative_raw.cpu().numpy(), query=query.cpu().numpy(),
        positive_query=positive_only.cpu().numpy(), positive_prompts=np.asarray(positive_prompts),
        negative_prompts=np.asarray(negative_prompts), class_names=np.asarray(classes))
    write_json(args.output / "query_fit.json", {"eta": .3, "support_count": len(support),
        "positive_counts": fit["positive_counts"].tolist(), "negative_counts": fit["negative_counts"].tolist(),
        "positive_prototype": "unit class-local positive support mean; unit(qbar + .3*unit(mean_positive))",
        "test_labels_used": False, "negative_text_encoder": "native loss.compute_text_features; genuine absence prompts"})

    cache_module_path = HISTORY / "code_phase6/vision_cache.py"
    expected_source = inputs["config"]["source_sha256"][str(cache_module_path)]
    if sha(cache_module_path) != expected_source:
        raise ValueError("Historical certified-cache reader source changed")
    module = load_module(cache_module_path, "_fewshot_radzero_visual_cache")
    token_manifest = inputs["tokens"]
    reader = module.VisionTokenCache(Path(inputs["token_info"]["path"]).parent,
        token_manifest["identity"], token_manifest["image_ids"], mode="read", prefetch=1)
    outputs = {name: np.asarray(values[:len(test)]).copy() for name, values in inputs["branch_arrays"].items()}
    positive_logits = np.empty((len(test), len(classes)), dtype=np.float32)
    maxima = {"original": 0.0, "query": 0.0}
    try:
        with torch.inference_mode():
            for start in range(0, len(test), args.batch_size):
                stop = min(start + args.batch_size, len(test))
                tokens = reader.get(start, stop)
                aligned = torch.from_numpy(np.array(tokens, copy=True)).to("cuda")
                vision = matcher.prepare_vision(aligned)
                for name, q in (("original", matcher.queries[0]), ("query", matcher.queries[1])):
                    current = matcher.match_prepared(vision, q)["logits"].cpu().numpy()
                    checked = assert_prediction_parity(current, outputs[name][start:stop], f"cached-token {name}")
                    maxima[name] = max(maxima[name], checked["max_abs_error"])
                positive_logits[start:stop] = matcher.match_prepared(vision, positive_only)["logits"].cpu().numpy()
                if start == 0:
                    n = min(2, stop)
                    metadata = inputs["metadata"]["test"]
                    # ViT FP32 matrix kernels can depend on batch size. Audit
                    # the exact historical encoder batch, even in a 4-image
                    # smoke run; using only two images here changes the path.
                    audit_count = min(int(token_manifest["identity"]["batch_size"]), len(test_all))
                    audit_tokens = reader.get(0, audit_count)
                    images = TestImages(metadata["image_paths"][:audit_count], Path(metadata["data_root"]), processor,
                                        dicom=inputs["dataset"] == "RSNA")
                    pixels = torch.stack([images[i][0] for i in range(audit_count)]).to("cuda")
                    fresh = model.forward_vision_model(pixels)["vision_tokens"]
                    parity["fresh_encoder"] = assert_prediction_parity(fresh.cpu().numpy(), audit_tokens, "fresh encoder / certified tokens")
                    parity["fresh_encoder"].update(images_checked=audit_count,
                        historical_batch_size=int(token_manifest["identity"]["batch_size"]))
                    local, reconstructed, _ = pool_original_matching(matcher.loss, positive_raw, aligned[:n])
                    parity["class_local"] = assert_prediction_parity(local.cpu().numpy(), arrays["test"]["class_local"][:n], "class-local cache")
                    parity["reconstructed_logits"] = assert_prediction_parity(reconstructed.cpu().numpy(), arrays["test"]["base_logits"][:n], "original cached logits")
                    parity["global_patch_mean"] = assert_prediction_parity(aligned[:n, 1:].mean(1).cpu().numpy(),
                        arrays["test"]["global_aligned"][:n, 1], "global aligned patch mean")
                    parity["author_api"] = positive_encoder.verify_reference(pixels[:n])
                if start == 0 or (start // args.batch_size) % 25 == 0 or stop == len(test):
                    progress(args, "RadZero", "matching_certified_tokens", stop, len(test))
    finally:
        reader.close()
    outputs["positive_query"] = positive_logits
    for name in ("original", "query"):
        parity[name] = {"passed": True, "images_checked": len(test), "max_abs_error": maxima[name],
                        "atol": 2e-5, "rtol": 2e-5}
    for name in ("view", "query_view"):
        parity[name] = {"passed": True, "reuse": "immutable complete phase6 logits; file SHA verified",
                        "images_checked": len(test)}
    # Double sigmoid preserves the ranking of finite FP32 native logits. The
    # historical RadZero CSV writer likewise uses torch.sigmoid(...double()).
    probabilities = {name: torch.sigmoid(torch.from_numpy(values).double()).numpy()
                     for name, values in outputs.items()}
    np.savez_compressed(args.output / "native_predictions.npz", **identity_arrays(test, classes), **probabilities)
    write_json(args.output / "native_parity.json", parity)
    write_json(args.output / "cache_provenance.json", inputs["provenance"])
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Frozen RadZero unexpectedly has trainable parameters")
    result = {"status": "smoke_complete" if args.limit else "complete", "model": "RadZero",
        "dataset": protocol["dataset"], "support_count": len(support), "test_count": len(test),
        "full_test_count": len(test_all), "class_names": classes, "feature_dimension": 768,
        "feature_definition": "unit raw aligned spatial patch mean excluding CLS; native query-conditioned unit class-local context",
        "logits_definition": "released RadZero positive-query matching logit divided by native learned temperature",
        "score_domain": "float64 sigmoid of native logits; views were averaged in native logit space",
        "native_maps": "not exported in this cache; classifier-only comparators have no adapted spatial head",
        "query_norm_policy": "unit(qbar + eta*unit(direction)); original eta0 raw query preserved",
        "test_labels_used": False, "selection": "fixed controls; method selection is a separate support-CV worker",
        "four_branch_reuse": "SHA-authenticated frozen phase6 arrays; original and query rechecked on every token batch",
        "positive_query_execution": "fresh native matching over all certified original-view visual tokens",
        "protocol_sha256": sha(args.protocol), "checkpoint": inputs["manifest"]["checkpoint"],
        "device": torch.cuda.get_device_name(0), "physical_gpu": args.gpu,
        "elapsed_seconds": time.time() - started, "native_parity": parity,
        "source_sha256": {str(path): sha(path) for path in [Path(__file__),
            Path(__file__).with_name("cache_native_common.py"), SETUP / "infer_batch.py",
            SETUP / "overnight_query_refinement.py", SETUP / "extract_frozen_features.py", cache_module_path]}}
    write_json(args.output / "result.json", result)
    write_json(args.output / "status.json", {"status": result["status"], "model": "RadZero",
        "dataset": protocol["dataset"], "done": len(test), "total": len(test_all)})
    print(json.dumps({key: result[key] for key in ("status", "model", "dataset", "test_count", "elapsed_seconds")}), flush=True)


def main():
    args = parse_arguments(__doc__, default_batch_size=16)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
