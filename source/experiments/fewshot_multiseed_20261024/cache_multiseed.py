#!/usr/bin/env python3
"""Export five support-only feature caches with authenticated native controls.

The CPU preparation step freezes old artifacts and input bytes. GPU workers read
only label-free test identities; test labels/boxes never enter fitting. Support
union images are encoded once. First-round test features and the four completed
stability arms are reused after checksum, identity, source and live parity checks.
Only the positive-prototype arm is newly evaluated for all five support draws.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
OLD = MEDICAL / "experiments/fewshot_native_query"
STABILITY = MEDICAL / "experiments/rsna_chest_stability_20261024"
STABLE_RUN = STABILITY / "runs/full_20261024_v1"
ALPHA = MEDICAL / "AlphaRAD"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("BiomedCLIP", "AlphaRAD")
DATASETS = ("rsna_pneumonia", "chestxdet10")
ALIASES = dict(original="original", query="query_refined", view="tta_center_original",
               query_view="tta_center_refined")
sys.path.insert(0, str(OLD))
from cache_native_common import (sha, write_json, load_module, save_features,
                                 identity_arrays, assert_prediction_parity, unit)


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, digest):
    path = Path(path).resolve()
    if not digest or sha(path) != digest:
        raise ValueError("Frozen source checksum changed: " + str(path))
    return path


def json_spec(spec, root):
    path = Path(spec["path"])
    if not path.is_absolute():
        path = root / path
    return checked(path, spec["sha256"])


def normalized_support(row):
    return dict(image=str(Path(row.get("image") or row["image_path"]).resolve()),
        labels=list(row["labels"]), group=str(row.get("group") or row.get("group_id") or ""),
        patient_id=str(row.get("patient_id") or ""), byte_sha256=row.get("byte_sha256"))


def source_files(model):
    if model == "BiomedCLIP":
        root = MEDICAL / "baselines/BiomedCLIP"
        files = [root / "local_setup/adapter.py",
            MEDICAL / "baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/adapter.py",
            MEDICAL / "baselines/local_setup/query_refinement_common.py",
            root / "checkpoints/open_clip_pytorch_model.bin"]
        files += sorted((root / "checkpoints").rglob("*.json"))
        files += sorted((root / "checkpoints").rglob("*.txt"))
        return files
    return [CHECKPOINT, ALPHA / "local_setup/query_refinement.py",
        ALPHA / "local_setup/eval_query_refinement.py", PHASE6 / "view_geometry.py",
        PHASE6 / "vision_cache.py"] + sorted((ALPHA / "cxrclip").rglob("*.py"))


def prepare_reuse(path):
    """CPU-only immutable snapshot, run before any campaign GPU worker starts."""
    import numpy as np
    if path.exists():
        raise FileExistsError("Do not overwrite frozen reuse provenance: " + str(path))
    snapshot = dict(version="fewshot-multiseed-reuse-v1", models={}, datasets={}, units={})
    for model in MODELS:
        snapshot["models"][model] = {str(p): sha(p) for p in source_files(model)}
    if snapshot["models"]["AlphaRAD"][str(CHECKPOINT)] != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint changed")
    for dataset in DATASETS:
        source = STABILITY / "protocol_v1" / (dataset + ".json")
        protocol = read(source)
        # Preparation discards labels and boxes. This label-free identity table
        # is the only test manifest opened by the subsequent model workers.
        identities = []
        for row in protocol["test_rows"]:
            image = str(Path(row["image"]).resolve())
            digest = sha(image)
            if digest != row["byte_sha256"]:
                raise ValueError("Frozen test image bytes changed: " + image)
            identities.append(dict(image=image, byte_sha256=digest,
                group=str(row["group_id"]), patient_id=str(row.get("patient_id") or "")))
        snapshot["datasets"][dataset] = dict(source=str(source), source_sha256=sha(source),
            class_names=protocol["class_names"], test_identities=identities)
        for model in MODELS:
            first = OLD / "runs" / ("full_biomed_20261024_v1" if model == "BiomedCLIP"
                else "full_alpha_20261024_v1") / f"{model}_{dataset}" / "cache"
            stable = STABLE_RUN / f"{model}_{dataset}"
            first_names = ["result.json", "test.npz", "queries.npz"]
            stable_names = ["result.json", "source_sha256.json", "support_union.json",
                            "support_union_features.npy"]
            for seed in SEEDS:
                stable_names.extend(f"seed_{seed}/{name}" for name in
                    ("result.json", "support_manifest.json", "query_fit.npz", "predictions.npz"))
                if model == "AlphaRAD" and dataset == "chestxdet10":
                    stable_names.append(f"seed_{seed}/native_maps.npz")
            first_record, stable_record = read(first / "result.json"), read(stable / "result.json")
            if first_record["status"] != "complete" or stable_record["status"] != "complete":
                raise ValueError("Only completed full-cohort caches are reusable")
            sources = {**first_record["source_sha256"], **read(stable / "source_sha256.json")}
            for filename, digest in sources.items():
                checked(filename, digest)
            for seed in SEEDS:
                checked(stable / f"seed_{seed}/predictions.npz",
                        stable_record["seed_results"][str(seed)]["prediction_sha256"])
            with np.load(first / "test.npz", allow_pickle=False) as values:
                if values["image_paths"].tolist() != [r["image"] for r in identities]:
                    raise ValueError("First-round test order differs from stability cohort")
                if "labels" in values.files:
                    raise ValueError("Test feature cache contains labels")
            record = dict(first_round=dict(path=str(first), files={n: sha(first / n) for n in first_names}),
                stability=dict(path=str(stable), files={n: sha(stable / n) for n in stable_names}),
                historical_sources=sources)
            if model == "AlphaRAD":
                info = first_record["vision_cache"]
                manifest = checked(Path(info["path"]) / "manifest.json", info["manifest_sha256"])
                token_manifest = read(manifest)
                if token_manifest["status"] != "complete":
                    raise ValueError("Token cache incomplete")
                record["vision_manifest"] = dict(path=str(manifest), sha256=sha(manifest))
            snapshot["units"][f"{model}_{dataset}"] = record
            print("Frozen reusable artifacts:", model, dataset, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, snapshot)
    print(json.dumps(dict(path=str(path), sha256=sha(path), status="frozen")), flush=True)


def load_job(args):
    """Authenticate label-free test data and exact per-seed support identities."""
    import numpy as np
    snapshot = read(args.reuse_manifest)
    if snapshot.get("version") != "fewshot-multiseed-reuse-v1":
        raise ValueError("Unknown reuse snapshot format")
    sources = {str(args.protocol): sha(args.protocol), str(args.reuse_manifest): sha(args.reuse_manifest),
               str(Path(__file__).resolve()): sha(__file__)}
    catalog = read(args.protocol)
    entry = catalog["datasets"][args.dataset]
    # Hash the multiseed dataset without opening its test label fields.
    if "path" in entry:
        path = json_spec(entry, args.protocol.parent)
        sources[str(path)] = sha(path)
    data = snapshot["datasets"][args.dataset]
    checked(data["source"], data["source_sha256"])
    classes = data["class_names"]
    test_all = data["test_identities"]
    expected_count = 5327 if args.dataset == "rsna_pneumonia" else 492
    if len(test_all) != expected_count or len({r["image"] for r in test_all}) != expected_count:
        raise ValueError("Fixed test cohort changed")
    for row in test_all:
        checked(row["image"], row["byte_sha256"])
    unit_record = snapshot["units"][f"{args.model}_{args.dataset}"]
    for block in ("first_round", "stability"):
        spec = unit_record[block]
        for name, digest in spec["files"].items():
            filename = checked(Path(spec["path"]) / name, digest)
            sources[str(filename)] = digest
    for filename, digest in {**snapshot["models"][args.model], **unit_record["historical_sources"]}.items():
        checked(filename, digest)
        sources[filename] = digest
    first, stable = (Path(unit_record[key]["path"]) for key in ("first_round", "stability"))
    first_record, stable_record = read(first / "result.json"), read(stable / "result.json")
    for record in (first_record, stable_record):
        if record["model"] != args.model or record["dataset"] != args.dataset or record["status"] != "complete":
            raise ValueError("Reused result model/dataset/completion mismatch")
    if stable_record["seeds"] != list(SEEDS) or stable_record["rows"] != len(test_all):
        raise ValueError("Reused stability scope changed")
    supports, protocols, union = {}, {}, {}
    for seed in SEEDS:
        spec = entry["seeds"][str(seed)]
        spec = spec.get("native_protocol", spec)
        path = json_spec(spec, args.protocol.parent)
        sources[str(path)] = sha(path)
        config = read(path)
        if config["class_names"] != classes or config["test_images"] != [r["image"] for r in test_all]:
            raise ValueError("New protocol changed classes or fixed test order")
        if config["dataset"] != args.dataset or float(config.get("eta", .3)) != .3:
            raise ValueError("Dataset/query strength changed")
        if config.get("support_seed") != seed or float(config.get("crop_fraction", .8)) != .8:
            raise ValueError("Support seed or fixed center crop changed")
        if (config["test_groups"] != [r["group"] for r in test_all] or
            config["test_patient_ids"] != [r["patient_id"] for r in test_all]):
            raise ValueError("New protocol test group/patient identity changed")
        rows = [normalized_support(row) for row in config["support_rows"]]
        old_rows = [normalized_support(row) for row in read(stable / f"seed_{seed}/support_manifest.json")["support_rows"]]
        if rows != old_rows:
            raise ValueError("Support identity/labels/order differ from same stability seed")
        if len(rows) != (10 if args.dataset == "rsna_pneumonia" else 50):
            raise ValueError("Unexpected support count")
        if len({r["group"] for r in rows}) != len(rows):
            raise ValueError("Duplicate support group")
        if config["support_groups"] != [r["group"] for r in rows]:
            raise ValueError("Support folds refer to incorrect groups")
        if {r["group"] for r in rows} & {r["group"] for r in test_all}:
            raise ValueError("Support/test group overlap")
        for row in rows:
            checked(row["image"], row["byte_sha256"])
            if row["image"] in union and row != union[row["image"]]:
                raise ValueError("Cross-seed metadata disagreement")
            union[row["image"]] = row
        supports[seed], protocols[seed] = rows, dict(path=str(path), sha256=sha(path), config=config)
    union = [union[key] for key in sorted(union)]
    if union != [normalized_support(row) for row in read(stable / "support_union.json")]:
        raise ValueError("Support union differs from stability cache")
    with np.load(first / "test.npz", allow_pickle=False) as arrays:
        if arrays["image_paths"].tolist() != [r["image"] for r in test_all] or arrays["class_names"].tolist() != classes:
            raise ValueError("First-round feature identity mismatch")
        if "labels" in arrays.files:
            raise ValueError("Test cache must not contain labels")
        test_arrays = {k: arrays[k][:args.smoke or None] for k in ("features", "local", "logits")}
    with np.load(first / "queries.npz", allow_pickle=False) as arrays:
        text = {k: arrays[k] for k in ("positive", "negative", "positive_raw", "negative_raw",
                                      "positive_prompts", "negative_prompts", "class_names")}
    if text["class_names"].tolist() != classes:
        raise ValueError("Reused text class order differs")
    rows = test_all[:args.smoke] if args.smoke else test_all
    predictions = {}
    for seed in SEEDS:
        with np.load(stable / f"seed_{seed}/predictions.npz", allow_pickle=False) as values:
            # Deliberately never access values['labels'].
            if values["image_paths"].tolist() != [r["image"] for r in test_all] or values["class_names"].tolist() != classes:
                raise ValueError("Stability prediction identity mismatch")
            if values["group_ids"].tolist() != [r["group"] for r in test_all]:
                raise ValueError("Stability test groups changed")
            if int(values["seed"]) != seed:
                raise ValueError("Stability prediction seed mismatch")
            predictions[seed] = {name: values[old][:len(rows)] for name, old in ALIASES.items()}
        for name in ("original", "view"):
            np.testing.assert_array_equal(predictions[seed][name], predictions[SEEDS[0]][name])
    return dict(classes=classes, supports=supports, union=union, protocols=protocols,
        test=rows, test_all=test_all, test_arrays=test_arrays, text=text, predictions=predictions,
        sources=sources, reuse=unit_record, first=first, stable=stable, first_record=first_record,
        stable_record=stable_record)


def progress(args, stage, started, done=0, total=0):
    value = dict(status=stage, model=args.model, dataset=args.dataset, done=done, total=total,
        elapsed_seconds=time.time()-started, smoke=args.smoke, seeds=list(SEEDS))
    write_json(args.output / "status.json", value)
    print(json.dumps(value), flush=True)


def parity(actual, expected, name):
    return assert_prediction_parity(actual, expected, name)


def verify_fit(job, seed, fit, indices):
    import numpy as np
    with np.load(job["stable"] / f"seed_{seed}/query_fit.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(indices, old["support_union_indices"])
        if set(fit) != set(old.files) - {"support_union_indices"}:
            raise ValueError("Native query-fit fields changed")
        checks = {}
        for key, actual in fit.items():
            if np.asarray(actual).dtype.kind in "biu":
                np.testing.assert_array_equal(actual, old[key])
                checks[key] = dict(passed=True, bitwise_equal=True)
            else:
                checks[key] = parity(actual, old[key], f"seed {seed} query-fit {key}")
        return checks


def export_support(args, job, support_arrays, positive, negative, positive_prompts, negative_prompts):
    import numpy as np
    import torch
    from torch.nn import functional as F
    lookup = {r["image"]: i for i, r in enumerate(job["union"])}
    queries, checks = {}, {}
    for seed in SEEDS:
        rows = job["supports"][seed]
        indices = np.asarray([lookup[r["image"]] for r in rows])
        arrays = {k: v[indices] for k, v in support_arrays.items()}
        labels = np.asarray([r["labels"] for r in rows], dtype=np.uint8)
        if args.model == "BiomedCLIP":
            from query_refinement_common import fit_directions, refine_query
            fit = fit_directions(arrays["features"], labels)
            q = refine_query(positive, fit["directions"], .3, preserve_norm=True)
            pq = refine_query(positive, F.normalize(fit["positive_means"], dim=-1), .3, preserve_norm=True)
            fit = {k: v.cpu().numpy() for k, v in fit.items()}
            fit.update(original_queries=positive.cpu().numpy(), original_negative_queries=negative.cpu().numpy(),
                       queries=q.cpu().numpy())
        else:
            from query_refinement import fit_query_directions
            fit = fit_query_directions(arrays["local"], labels, positive.cpu().numpy(), etas=(0., .3))
            q = torch.as_tensor(fit["queries"][1], device=positive.device)
            pq = F.normalize(F.normalize(positive, dim=-1) + .3 *
                F.normalize(torch.as_tensor(fit["positive_means"], device=positive.device), dim=-1), dim=-1)
        checks[str(seed)] = verify_fit(job, seed, fit, indices)
        folder = args.output / f"seed_{seed}"
        save_features(folder / "support.npz", rows, job["classes"], arrays, support=True)
        np.savez_compressed(folder / "query_fit.npz", **fit, support_union_indices=indices)
        normalized_positive = positive.cpu().numpy() if args.model == "BiomedCLIP" else unit(positive.cpu().numpy())
        normalized_negative = negative.cpu().numpy() if args.model == "BiomedCLIP" else unit(negative.cpu().numpy())
        np.savez_compressed(folder / "queries.npz", positive=normalized_positive,
            negative=normalized_negative, positive_raw=positive.cpu().numpy(),
            negative_raw=negative.cpu().numpy(), query=q.cpu().numpy(), positive_query=pq.cpu().numpy(),
            positive_prompts=np.asarray(positive_prompts), negative_prompts=np.asarray(negative_prompts),
            class_names=np.asarray(job["classes"]))
        write_json(folder / "query_fit.json", dict(eta=.3, support_count=len(rows),
            positive_counts=fit["positive_counts"].tolist(), negative_counts=fit["negative_counts"].tolist(),
            test_labels_used=False, positive_query="recomputed from this seed's own positive support means",
            stability_query_fit_parity=checks[str(seed)]))
        queries[seed] = dict(query=q, positive_query=pq)
    return queries, checks


def biomed(args, job, started):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(MEDICAL / "baselines/local_setup"))
    from query_refinement_common import center_pixels
    path = MEDICAL / "baselines/protocols/alpharad_binary/BiomedCLIP/local_setup/adapter.py"
    native = load_module(path, "_multiseed_biomed_native").create_adapter("cuda")
    native.model.requires_grad_(False).eval()
    old_metadata = job["first_record"]["adapter_metadata"]
    strip_address = lambda x: re.sub(r"0x[0-9a-fA-F]+", "0xADDRESS", x)
    if strip_address(native.metadata["preprocess"]) != strip_address(old_metadata["preprocess"]):
        raise ValueError("Native BiomedCLIP preprocessing changed")
    for key in ("checkpoint", "precision", "protocol_revision", "temperature", "logit_scale_used"):
        if native.metadata[key] != old_metadata[key]:
            raise ValueError("Native BiomedCLIP model/scoring identity changed: " + key)
    count = len(job["classes"])
    phrases = [name.replace("_", " ") for name in job["classes"]]
    neg_prompts, pos_prompts = [f"There is no {s}." for s in phrases], [f"There is {s}." for s in phrases]
    with torch.inference_mode():
        text = native.model.encode_text(native.tokenizer(neg_prompts + pos_prompts).to(native.device), normalize=True)
    negative, positive = text[:count], text[count:]
    checks = {k: parity(v.cpu().numpy(), job["text"][k], "native text " + k)
              for k, v in (("positive_raw", positive), ("negative_raw", negative))}
    def pixels(rows):
        batch = []
        for row in rows:
            with Image.open(row["image"]) as image:
                batch.append(native.preprocess(image.convert("RGB")))
        return torch.stack(batch).to(native.device)
    def score(features, pos):
        similarity = features @ torch.cat((negative, pos)).T
        neg, yes = similarity[:, :count], similarity[:, count:]
        return torch.stack((neg, yes), -1).softmax(-1)[..., 1], yes-neg
    parts = []
    with torch.inference_mode():
        for offset in range(0, len(job["union"]), args.batch_size):
            parts.append(native.model.encode_image(pixels(job["union"][offset:offset+args.batch_size]), normalize=True).float().cpu().numpy())
            progress(args, "support_union", started, min(offset+args.batch_size, len(job["union"])), len(job["union"]))
        feature = np.concatenate(parts)
        support = dict(features=feature, local=np.broadcast_to(feature[:, None], (len(feature), count, feature.shape[-1])).copy(),
            logits=score(torch.as_tensor(feature, device=native.device), positive)[1].cpu().numpy())
        old_support = np.load(job["stable"] / "support_union_features.npy", allow_pickle=False)
        checks["support_union"] = parity(feature, old_support, "support union features")
        queries, checks["query_fit"] = export_support(args, job, support, positive, negative, pos_prompts, neg_prompts)
        # Live first-batch original/center forwards authenticate preprocessing,
        # all feature fields, and all four reused arms for every support seed.
        first_rows = job["test"][:args.batch_size]
        x = pixels(first_rows)
        fresh = native.model.encode_image(x, normalize=True).float()
        center = native.model.encode_image(center_pixels(x, "center80"), normalize=True).float()
        original, logits = score(fresh, positive)
        local = fresh[:, None].expand(-1, count, -1)
        checks["live_features"] = {k: parity(v.cpu().numpy(), job["test_arrays"][k][:len(first_rows)], "live " + k)
            for k, v in (("features", fresh), ("local", local), ("logits", logits))}
        checks["live_native_arms"] = {}
        for seed in SEEDS:
            q = score(fresh, queries[seed]["query"])[0]
            arms = dict(original=original, query=q, view=(original+score(center, positive)[0])*.5,
                        query_view=(q+score(center, queries[seed]["query"])[0])*.5)
            checks["live_native_arms"][str(seed)] = {k: parity(v.cpu().numpy(), job["predictions"][seed][k][:len(first_rows)],
                f"seed {seed} live native {k}") for k, v in arms.items()}
        for seed in SEEDS:
            parts = []
            for offset in range(0, len(job["test"]), 512):
                features = torch.as_tensor(job["test_arrays"]["features"][offset:offset+512], device=native.device)
                parts.append(score(features, queries[seed]["positive_query"])[0].cpu().numpy())
                checks.setdefault("original_all_images", {})[str(offset)] = parity(score(features, positive)[0].cpu().numpy(),
                    job["predictions"][seed]["original"][offset:offset+512], "cached features native original")
            job["predictions"][seed]["positive_query"] = np.concatenate(parts)
    if any(p.requires_grad for p in native.model.parameters()):
        raise ValueError("Backbone unexpectedly trainable")
    return checks, {}, dict(adapter_metadata=native.metadata, image_test_encoding="live first batch only; authenticated shared features thereafter")


def alpha(args, job, started):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.nn import functional as F
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.evaluator_utils import disease_name_mapping
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator
    from view_geometry import apply_view, aggregate_outputs
    from vision_cache import VisionTokenCache, image_fingerprints
    from cache_alpharad import write_images
    seed_everything(1234)
    for name, rows in (("support_union", job["union"]), ("test", job["test"])):
        write_images(args.output / f"{name}_inputs.csv", rows)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(CHECKPOINT)]},
        "data_test": {name: dict(name=name, data_type="image_classification",
            data_path=str(args.output / f"{name}_inputs.csv"), n_class=len(job["classes"]), normalize=None)
            for name in ("support_union", "test")}, "dataloader": {"test": options}}
    write_json(args.output / "eval_config.json", config)
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    positive_raw, pos_prompts = original_queries_from_evaluator(evaluator, job["classes"])
    neg_prompts = [f"There is no {disease_name_mapping(name)}." for name in job["classes"]]
    negative_raw = np.asarray(evaluator.encode_text(neg_prompts,
        text_max_length=evaluator.config["base"]["dqn_label_max_length"])["cls_emb_raw"], dtype=np.float32).reshape(len(job["classes"]), -1)
    matcher = QueryRefinementMatcher(evaluator.model, positive_raw[None], (0.,), use_all_image_tokens=evaluator.use_all_image_tokens)
    positive = torch.as_tensor(positive_raw, device=matcher.device)
    negative = torch.as_tensor(negative_raw, device=matcher.device)
    checks = {k: parity(v, job["text"][k], "native text " + k)
              for k, v in (("positive_raw", positive_raw), ("negative_raw", negative_raw))}
    def features(vision, original):
        if len(vision["image_patch_raw_features"]) != 1:
            raise ValueError("Expected published single projection layer")
        return dict(features=F.normalize(vision["image_patch_raw_features"][0].mean(1), dim=-1),
                    local=original["class_local"], logits=original["logits"])
    parts, observed = {k: [] for k in ("features", "local", "logits")}, []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support_union"]:
            vision = matcher.encode_vision(batch["images"])
            output = matcher.match_prepared(vision, positive_raw, return_features=True)
            for key, value in features(vision, output).items():
                parts[key].append(value.cpu().numpy())
            observed.extend(batch["image_paths"])
            progress(args, "support_union", started, len(observed), len(job["union"]))
    if observed != [r["image"] for r in job["union"]]:
        raise ValueError("Support union loader changed order")
    support = {k: np.concatenate(v) for k, v in parts.items()}
    checks["support_union"] = parity(support["local"], np.load(job["stable"] / "support_union_features.npy", allow_pickle=False), "support union local")
    queries, checks["query_fit"] = export_support(args, job, support, positive, negative, pos_prompts, neg_prompts)
    manifest_path = json_spec(job["reuse"]["vision_manifest"], args.reuse_manifest.parent)
    manifest = read(manifest_path)
    identity = manifest["identity"]
    paths = [r["image"] for r in job["test_all"]]
    if (manifest["status"] != "complete" or manifest["image_ids"] != paths or
        identity["checkpoint_sha256"] != CHECKPOINT_SHA or
        identity["use_all_image_tokens"] != evaluator.use_all_image_tokens or
        identity["num_heads"] != evaluator.model.num_radzero_heads or
        identity["input_file_fingerprints"] != image_fingerprints(paths)):
        raise ValueError("Certified AlphaRAD token identity changed")
    cache = VisionTokenCache(manifest_path.parent.parent, identity, paths, mode="read", prefetch=1)
    cache_record = dict(active=True, path=str(manifest_path.parent), manifest_sha256=sha(manifest_path), mode="read_only")
    spatial_paths = []
    if args.dataset == "chestxdet10":
        with np.load(job["stable"] / "seed_42/native_maps.npz", allow_pickle=False) as old:
            spatial_paths = old["image_paths"].tolist()
        if len(spatial_paths) != 409 or len(set(spatial_paths)) != 409 or not set(spatial_paths) <= set(paths):
            raise ValueError("Fixed spatial cohort changed")
    spatial_set = set(spatial_paths)
    spatial_observed = []
    map_parts = {s: [] for s in SEEDS}
    pred_parts = {s: [] for s in SEEDS}
    checks["original_all_images"], checks["query_all_images"], checks["live_native_arms"] = {}, {}, {}
    offset = 0
    try:
        with torch.inference_mode():
            for batch in evaluator.data_loader_dict["test"]:
                stop = offset + len(batch["images"])
                if batch["image_paths"] != paths[offset:stop]:
                    raise ValueError("Test loader order changed")
                if cache is not None:
                    tokens = torch.as_tensor(np.array(cache.get(offset, stop), copy=True), dtype=torch.float32, device=matcher.device)
                    vision = dict(image_cls_raw_features=[tokens[:, 0]], image_patch_raw_features=[tokens[:, 1:]])
                else:
                    vision = matcher.encode_vision(batch["images"])
                if offset == 0:
                    fresh = matcher.encode_vision(batch["images"])
                    try:
                        for key in vision:
                            torch.testing.assert_close(vision[key][0], fresh[key][0], atol=2e-5, rtol=1e-5)
                    except AssertionError:
                        cache.close()
                        cache = None
                        cache_record.update(active=False, first_batch_fresh_token_check=False,
                            fallback="Fresh original-view encoding for all test images; strict token parity was not relaxed")
                        vision = fresh
                    else:
                        cache_record["first_batch_fresh_token_check"] = True
                    fresh_original = matcher.match_prepared(fresh, positive_raw, return_features=True)
                    checks["live_features"] = {key: parity(value.cpu().numpy(), job["test_arrays"][key][:stop], "live " + key)
                        for key, value in features(fresh, fresh_original).items()}
                    center = matcher.encode_vision(apply_view(batch["images"], "center80"))
                    center_original = matcher.match_prepared(center, positive_raw)
                    for seed in SEEDS:
                        q = matcher.match_prepared(fresh, queries[seed]["query"])
                        cq = matcher.match_prepared(center, queries[seed]["query"])
                        arms = dict(original=fresh_original, query=q,
                            view=aggregate_outputs(dict(identity=fresh_original, center80=center_original), ["identity", "center80"]),
                            query_view=aggregate_outputs(dict(identity=q, center80=cq), ["identity", "center80"]))
                        checks["live_native_arms"][str(seed)] = {key: parity(torch.sigmoid(value["logits"]).cpu().numpy(),
                            job["predictions"][seed][key][:stop], f"seed {seed} live {key}") for key, value in arms.items()}
                selected = [j for j, path in enumerate(batch["image_paths"]) if path in spatial_set]
                spatial_observed.extend(batch["image_paths"][j] for j in selected)
                original = matcher.match_prepared(vision, positive_raw)
                checks["original_all_images"][str(offset)] = parity(torch.sigmoid(original["logits"]).cpu().numpy(),
                    job["predictions"][SEEDS[0]]["original"][offset:stop], "token-cache original")
                for seed in SEEDS:
                    q = matcher.match_prepared(vision, queries[seed]["query"])
                    checks["query_all_images"].setdefault(str(seed), {})[str(offset)] = parity(torch.sigmoid(q["logits"]).cpu().numpy(),
                        job["predictions"][seed]["query"][offset:stop], f"seed {seed} token-cache Q")
                    pq = matcher.match_prepared(vision, queries[seed]["positive_query"], return_maps=bool(selected))
                    pred_parts[seed].append(torch.sigmoid(pq["logits"]).cpu().numpy())
                    if selected:
                        map_parts[seed].append(pq["grounding_map"][selected].cpu().numpy())
                offset = stop
                if offset == len(job["test"]) or offset // args.batch_size % 25 == 1:
                    progress(args, "positive_query_test", started, offset, len(job["test"]))
    finally:
        if cache is not None:
            cache.close()
    for seed in SEEDS:
        job["predictions"][seed]["positive_query"] = np.concatenate(pred_parts[seed])
    maps = {}
    if spatial_observed:
        if spatial_observed != [p for p in spatial_paths if p in set(paths[:len(job["test"])])]:
            raise ValueError("Spatial map order changed")
        for seed in SEEDS:
            with np.load(job["stable"] / f"seed_{seed}/native_maps.npz", allow_pickle=False) as old:
                if old["image_paths"].tolist() != spatial_paths or old["class_names"].tolist() != job["classes"]:
                    raise ValueError("Reused spatial map identities changed")
                indices = [spatial_paths.index(p) for p in spatial_observed]
                joined = {key: old[old_key][indices] for key, old_key in ALIASES.items()}
            joined["positive_query"] = np.concatenate(map_parts[seed])
            maps[seed] = dict(image_paths=np.asarray(spatial_observed), class_names=np.asarray(job["classes"]), **joined)
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Backbone unexpectedly trainable")
    return checks, maps, dict(checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored, vision_cache=cache_record,
        preprocessing=OmegaConf.to_container(OmegaConf.create(evaluator.config["transform"]), resolve=True),
        image_test_encoding="certified original tokens; fresh first identity/center batch parity")


def run(args):
    import numpy as np
    started = time.time()
    job = load_job(args)
    if args.audit_only:
        print(json.dumps(dict(status="audit_passed", model=args.model, dataset=args.dataset,
            test_count=len(job["test"]), support_union_count=len(job["union"]), seeds=list(SEEDS))), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    progress(args, "inputs_authenticated", started)
    for seed in SEEDS:
        (args.output / f"seed_{seed}").mkdir()
    write_json(args.output / "reuse_audit.json", dict(reuse_manifest=str(args.reuse_manifest),
        reuse_manifest_sha256=sha(args.reuse_manifest), files=job["sources"], test_input_byte_sha256_verified=True,
        support_identities_match_stability=True, fixed_test_count=len(job["test_all"])))
    write_json(args.output / "support_union.json", job["union"])
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    progress(args, "loading", started)
    checks, maps, metadata = (biomed if args.model == "BiomedCLIP" else alpha)(args, job, started)
    save_features(args.output / "test.npz", job["test"], job["classes"], job["test_arrays"])
    status = "smoke_complete" if args.smoke else "complete"
    result_paths = {}
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        (folder / "test.npz").symlink_to("../test.npz")
        for key, values in job["predictions"][seed].items():
            if values.shape != (len(job["test"]), len(job["classes"])) or not np.isfinite(values).all():
                raise ValueError("Malformed native prediction " + key)
        np.savez_compressed(folder / "native_predictions.npz", **identity_arrays(job["test"], job["classes"]),
                            **job["predictions"][seed])
        if seed in maps:
            np.savez_compressed(folder / "native_maps.npz", **maps[seed])
        old = job["first_record"]
        result = dict(status=status, model=args.model, dataset=args.dataset, seed=seed,
            support_count=len(job["supports"][seed]), test_count=len(job["test"]), full_test_count=len(job["test_all"]),
            class_names=job["classes"], feature_dimension=job["test_arrays"]["features"].shape[-1],
            feature_definition=old["feature_definition"], logits_definition=old["logits_definition"],
            score_domain=old["score_domain"], query_norm_policy=old["query_norm_policy"],
            protocol_sha256=job["protocols"][seed]["sha256"], protocol=job["protocols"][seed]["path"],
            catalog_sha256=sha(args.protocol), reuse_manifest_sha256=sha(args.reuse_manifest),
            reuse_audit=str(args.output / "reuse_audit.json"), native_parity=str(args.output / "native_parity.json"),
            test_labels_used=False, test_boxes_used=False, no_gradient_training=True,
            support_union_encoded_once=True, positive_query_recomputed_from_current_seed=True,
            native_map_count=len(maps[seed]["image_paths"]) if seed in maps else 0,
            metadata=metadata, elapsed_seconds=time.time()-started)
        artifacts = ("support.npz", "test.npz", "queries.npz", "query_fit.npz", "native_predictions.npz")
        result["artifact_sha256"] = {name: sha(folder / name) for name in artifacts}
        if seed in maps:
            result["artifact_sha256"]["native_maps.npz"] = sha(folder / "native_maps.npz")
        write_json(folder / "result.json", result)
        write_json(folder / "status.json", dict(status=status, model=args.model, dataset=args.dataset, seed=seed))
        result_paths[str(seed)] = dict(path=str(folder / "result.json"), sha256=sha(folder / "result.json"))
    write_json(args.output / "native_parity.json", checks)
    # Detect changes to any authenticated source during the GPU work.
    for path, digest in job["sources"].items():
        checked(path, digest)
    write_json(args.output / "result.json", dict(status=status, model=args.model, dataset=args.dataset,
        seeds=list(SEEDS), seed_results=result_paths, test_count=len(job["test"]), full_test_count=len(job["test_all"]),
        support_union_count=len(job["union"]), protocol_sha256=sha(args.protocol),
        reuse_manifest=str(args.reuse_manifest), reuse_manifest_sha256=sha(args.reuse_manifest),
        source_sha256={str(Path(__file__).resolve()): sha(__file__)}, elapsed_seconds=time.time()-started,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_gpu=args.gpu, metadata=metadata,
        test_labels_used=False, all_seed_fits_match_stability=True, native_parity_sha256=sha(args.output / "native_parity.json")))
    progress(args, status, started, len(job["test"]), len(job["test"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--reuse-manifest", type=Path, default=HERE / "cache_reuse_sources_v1.json")
    parser.add_argument("--prepare-reuse", action="store_true", help="CPU only: freeze reusable artifacts once and exit")
    parser.add_argument("--audit-only", action="store_true", help="CPU only: verify sources/protocol/cache identities and exit")
    args = parser.parse_args()
    args.reuse_manifest = args.reuse_manifest.resolve()
    if args.prepare_reuse:
        prepare_reuse(args.reuse_manifest)
        return
    if not all((args.protocol, args.model, args.dataset)):
        parser.error("--protocol, --model and --dataset are required")
    if not args.audit_only and (args.output is None or args.gpu not in (4, 5, 6, 7)):
        parser.error("Inference requires a fresh --output and authorized physical --gpu 4..7")
    args.batch_size = args.batch_size or (16 if args.model == "BiomedCLIP" else 8)
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error("Invalid batch-size, workers or smoke")
    args.protocol = args.protocol.resolve()
    if args.output:
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError("Use a fresh cache output directory: " + str(args.output))
    # Preserve controller UUID binding rather than replacing it with an index.
    if args.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "4")
    try:
        run(args)
    except BaseException as error:
        if args.output and args.output.is_dir():
            write_json(args.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
            write_json(args.output / "status.json", dict(status="failed", error=repr(error), model=args.model, dataset=args.dataset))
        raise


if __name__ == "__main__":
    main()
