#!/usr/bin/env python3
"""Freeze BiomedCLIP, export shared features, and execute fixed native controls.

The official image/text encoders use the previously audited AlphaRAD binary
protocol: raw cosine presence-versus-absence softmax, temperature one. Test
identities, never test labels, enter this worker. No hyperparameter selection.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

from cache_native_common import (MEDICAL, BRANCHES, parse_arguments, load_protocol,
    load_module, write_json, sha, identity_arrays, save_features, progress,
    assert_prediction_parity)


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image

    sys.path.insert(0, str(MEDICAL / "baselines/local_setup"))
    from query_refinement_common import center_pixels, fit_directions, refine_query

    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device is unavailable")
    protocol, support, test_all = load_protocol(args.protocol)
    test = test_all[:args.limit] if args.limit else test_all
    classes = protocol["class_names"]
    count = len(classes)
    adapter_path = MEDICAL / "baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/adapter.py"
    progress(args, "BiomedCLIP", "loading")
    native = load_module(adapter_path, "_fewshot_native_biomed").create_adapter("cuda")
    native.model.eval().requires_grad_(False)
    phrases = [name.replace("_", " ") for name in classes]
    negative_prompts = [f"There is no {name}." for name in phrases]
    positive_prompts = [f"There is {name}." for name in phrases]
    with torch.inference_mode():
        text = native.model.encode_text(native.tokenizer(negative_prompts + positive_prompts).to(native.device), normalize=True)
    negative, positive = text[:count], text[count:]
    native._binary_text[tuple(phrases)] = text

    def pixels(rows):
        values = []
        for row in rows:
            with Image.open(row["image"]) as image:
                values.append(native.preprocess(image.convert("RGB")))
        return torch.stack(values).to(native.device)

    def score(features, pos):
        similarities = features @ torch.cat((negative, pos)).T
        neg, yes = similarities[:, :count], similarities[:, count:]
        probabilities = torch.stack((neg, yes), dim=-1).softmax(-1)[..., 1]
        return probabilities, yes - neg

    support_features, support_logits = [], []
    with torch.inference_mode():
        for offset in range(0, len(support), args.batch_size):
            rows = support[offset:offset + args.batch_size]
            features = native.model.encode_image(pixels(rows), normalize=True).float()
            support_features.append(features.cpu().numpy())
            support_logits.append(score(features, positive)[1].cpu().numpy())
    support_features = np.concatenate(support_features)
    support_labels = np.asarray([row["labels"] for row in support], dtype=np.uint8)
    fitted = fit_directions(support_features, support_labels)
    query = refine_query(positive, fitted["directions"], .3, preserve_norm=True)
    positive_direction = F.normalize(fitted["positive_means"], dim=-1)
    positive_query = refine_query(positive, positive_direction, .3, preserve_norm=True)
    save_features(args.output / "support.npz", support, classes, {
        "features": support_features,
        "local": np.broadcast_to(support_features[:, None], (len(support), count, support_features.shape[-1])).copy(),
        "logits": np.concatenate(support_logits)}, support=True)
    np.savez_compressed(args.output / "queries.npz", positive=positive.cpu().numpy(),
        negative=negative.cpu().numpy(), positive_raw=positive.cpu().numpy(),
        negative_raw=negative.cpu().numpy(), query=query.cpu().numpy(),
        positive_query=positive_query.cpu().numpy(),
        positive_prompts=np.asarray(positive_prompts), negative_prompts=np.asarray(negative_prompts),
        class_names=np.asarray(classes))
    write_json(args.output / "query_fit.json", {"eta": .3, "support_count": len(support),
        "positive_counts": fitted["positive_counts"].tolist(),
        "negative_counts": fitted["negative_counts"].tolist(),
        "positive_prototype": "unit positive support mean; eta residual; preserve original query norm",
        "test_labels_used": False})

    arrays = {name: [] for name in BRANCHES}
    features_out, logits_out = [], []
    parity = {}
    with torch.inference_mode():
        for offset in range(0, len(test), args.batch_size):
            rows = test[offset:offset + args.batch_size]
            x = pixels(rows)
            features = native.model.encode_image(x, normalize=True).float()
            centered = native.model.encode_image(center_pixels(x, "center80"), normalize=True).float()
            original, logits = score(features, positive)
            refined = score(features, query)[0]
            outputs = {"original": original, "query": refined,
                "view": .5 * (original + score(centered, positive)[0]),
                "query_view": .5 * (refined + score(centered, query)[0]),
                "positive_query": score(features, positive_query)[0]}
            if offset == 0:
                expected = native.predict([row["image"] for row in rows], classes)
                parity["native_adapter"] = assert_prediction_parity(original.cpu().numpy(), expected, "native binary adapter")
            for branch, values in outputs.items():
                arrays[branch].append(values.cpu().numpy())
            features_out.append(features.cpu().numpy())
            logits_out.append(logits.cpu().numpy())
            if offset == 0 or (offset // args.batch_size) % 25 == 0 or offset + len(rows) == len(test):
                progress(args, "BiomedCLIP", "test", offset + len(rows), len(test))
    arrays = {name: np.concatenate(parts).astype(np.float32) for name, parts in arrays.items()}
    features_out = np.concatenate(features_out)
    save_features(args.output / "test.npz", test, classes, {
        "features": features_out,
        "local": np.broadcast_to(features_out[:, None], (len(test), count, features_out.shape[-1])).copy(),
        "logits": np.concatenate(logits_out)})

    # Old predictions are used only as immutable implementation-parity checks.
    reference = protocol["references"]["BiomedCLIP"]
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Previous native result changed")
    with np.load(Path(reference["path"]).parent / "predictions.npz", allow_pickle=False) as old:
        if old["class_names"].tolist() != classes:
            raise ValueError("Old class order changed")
        lookup = {str(Path(path).resolve()): i for i, path in enumerate(old["image_paths"].tolist())}
        indices = [lookup[str(Path(row["image"]).resolve())] for row in test]
        for name, old_name in (("original", "original"), ("query", "query_refined"),
                              ("view", "tta_center_original"), ("query_view", "tta_center_refined")):
            parity[name] = assert_prediction_parity(arrays[name], old[old_name][indices], f"old BiomedCLIP {name}")
    np.savez_compressed(args.output / "native_predictions.npz", **identity_arrays(test, classes), **arrays)
    write_json(args.output / "native_parity.json", parity)
    if any(p.requires_grad for p in native.model.parameters()):
        raise ValueError("The backbone unexpectedly has trainable parameters")
    result = {"status": "smoke_complete" if args.limit else "complete", "model": "BiomedCLIP",
        "dataset": protocol["dataset"], "support_count": len(support), "test_count": len(test),
        "full_test_count": len(test_all), "class_names": classes, "feature_dimension": features_out.shape[-1],
        "feature_definition": "L2-normalized native projected global image; local is class broadcast of this same vector",
        "logits_definition": "native temperature-one positive cosine minus negative cosine",
        "score_domain": "native per-disease binary softmax probability; two views average probabilities",
        "native_maps": "unsupported; no verified native BiomedCLIP spatial evaluator",
        "query_norm_policy": "preserve original native positive-query norm", "test_labels_used": False,
        "selection": "fixed native controls only; support-only comparator selection is a separate worker",
        "protocol_sha256": sha(args.protocol), "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(0), "physical_gpu": args.gpu,
        "native_parity": parity, "adapter_metadata": native.metadata,
        "source_sha256": {str(path): sha(path) for path in [Path(__file__),
            Path(__file__).with_name("cache_native_common.py"), adapter_path,
            MEDICAL / "baselines/local_setup/query_refinement_common.py"]}}
    write_json(args.output / "result.json", result)
    write_json(args.output / "status.json", {"status": result["status"], "model": "BiomedCLIP",
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
