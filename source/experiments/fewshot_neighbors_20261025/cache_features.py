#!/usr/bin/env python3
"""Cache frozen CheXchoNet features and native single/equal-dual-view controls.

No comparator fitting or test metrics run here. Fit-visible test caches contain
no labels; the separate protocol label asset is reserved for post-fit scoring.
Each task/seed uses exactly its own ten inherited original-view supports.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STABILITY = ROOT / "experiments/chexchonet_stability_20261024"
OLD_RUN = STABILITY / "runs/full_20261024_v1"
ALPHA = ROOT / "AlphaRAD"
PHASE6 = ROOT / "experiments/query_search_20261021_152010/code_phase6"
for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np

spec = importlib.util.spec_from_file_location("neighbors_inherited_stability", STABILITY / "infer_multiseed.py")
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)
TASKS, SEEDS = stability.TASKS, stability.SEEDS
sha, write = stability.sha, stability.write
BRANCHES = ("original", "query", "original_view", "query_view")
OLD_BRANCHES = dict(original="original", query="query_refined", original_view="tta_center_original", query_view="tta_center_refined")


def load_protocol_without_test_labels(path):
    spec = importlib.util.spec_from_file_location("neighbors_cache_protocol", HERE / "prepare_protocol.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    protocol = module.load_protocol(path, include_test_labels=False)
    # Keep the small native-query fitting API compatible without exposing any
    # other task's labels to the task fit. Pool and test rows remain unlabelled.
    protocol["supports"] = {seed: {task: [dict(image_path=r["image_path"], patient_id=r["patient_id"],
        **{task: int(r["labels"][0])}) for r in protocol["supports"][seed][task]]
        for task in TASKS} for seed in SEEDS}
    for key in ("pool", "test"):
        if any(set(row) - {"image_path", "patient_id"} for row in protocol[key]):
            raise ValueError("Fit-facing candidate/test identities must not contain labels")
    register(protocol, [HERE / "prepare_protocol.py"])
    return protocol, module


def identities(rows, classes):
    return dict(image_paths=np.asarray([r["image_path"] for r in rows]),
                patient_ids=np.asarray([r["patient_id"] for r in rows]),
                class_names=np.asarray(classes), task_keys=np.asarray(TASKS))


def progress(args, stage, started, done=0, total=0):
    item = dict(status=stage, model=args.model, done=done, total=total,
                elapsed_seconds=time.time() - started, smoke=args.smoke)
    write(args.output / "status.json", item)
    print(json.dumps(item), flush=True)


def register(protocol, paths):
    for path in paths:
        path = Path(path).resolve()
        protocol["sources"][str(path)] = sha(path)


def support_indices(protocol, output):
    lookup = {r["image_path"]: i for i, r in enumerate(protocol["pool"])}
    selected = {}
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        folder.mkdir()
        selected[seed] = {}
        for task in TASKS:
            rows = protocol["supports"][seed][task]
            indices = np.asarray([lookup[r["image_path"]] for r in rows], dtype=np.int64)
            if len(indices) != 10 or len(set(indices.tolist())) != 10 or sum(r[task] for r in rows) != 5:
                raise ValueError("Unexpected task-specific support budget")
            selected[seed][task] = indices
        np.savez_compressed(folder / "support_indices.npz", **selected[seed])
        write(folder / "support_manifest.json", dict(seed=seed, task_support_rows=protocol["supports"][seed],
              support_view="identity", support_labels_used_only_for_own_task=True,
              comparator_view_policy="same original-support fit reused for both test-view conditions"))
    return selected


def compare(actual, expected, label, atol=2e-6, rtol=1e-5):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=label)
    return dict(passed=True, max_abs_error=float(np.max(np.abs(actual - expected))),
                bitwise_equal=bool(np.array_equal(actual, expected)), atol=atol, rtol=rtol)


def historical_checks(args, protocol, rows, arrays, pool_native, refined):
    """Compare fresh native endpoints with all previously fixed support draws."""
    old = OLD_RUN / args.model
    report = {}
    audit_path = old / "input_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["class_names"] != protocol["classes"] or audit["task_keys"] != list(TASKS):
        raise ValueError("Historical task identity differs")
    register(protocol, [audit_path])
    # These old feature arrays have no embedded row identities. The new protocol
    # preserves the old candidate-pool order and verifies its source bytes. Check
    # the old run's own source snapshot without opening its labelled test rows.
    old_protocol = Path(audit["protocol"])
    if sha(old_protocol) != audit["protocol_sha256"]:
        raise ValueError("Historical cache protocol changed")
    old_sources_path = old / "source_sha256.json"
    old_sources = json.loads(old_sources_path.read_text())
    source_pool = str(stability.POOL.resolve())
    if old_sources.get(source_pool) != sha(source_pool):
        raise ValueError("Historical candidate-pool source bytes changed")
    register(protocol, [old_protocol, old_sources_path, source_pool])
    feature_name = "candidate_class_local.npy" if args.model == "AlphaRAD" else "candidate_global_features.npy"
    feature_path = old / feature_name
    report["candidate_native_features"] = compare(pool_native, np.load(feature_path, allow_pickle=False),
                                                   "historical candidate features", atol=2e-5)
    register(protocol, [feature_path])
    for seed in SEEDS:
        old_support = json.loads((old / f"seed_{seed}" / "support_manifest.json").read_text())
        for task in TASKS:
            previous = old_support["task_support_rows"][task]
            current = protocol["supports"][seed][task]
            signature = lambda rs: [(r["image_path"], r["patient_id"], int(r[task])) for r in rs]
            if signature(previous) != signature(current):
                raise ValueError("Historical task/seed support changed")
            old_fit_path = old / f"seed_{seed}" / f"query_fit_{task}.npz"
            with np.load(old_fit_path, allow_pickle=False) as old_fit, np.load(args.output / f"seed_{seed}" / "support_indices.npz") as new_ids:
                np.testing.assert_array_equal(old_fit["support_indices"], new_ids[task])
            register(protocol, [old_fit_path])
        query_path = old / f"seed_{seed}" / "queries.npy"
        report[f"seed_{seed}/queries"] = compare(refined[seed], np.load(query_path, allow_pickle=False), "historical query")
        path = old / f"seed_{seed}" / "predictions.npz"
        with np.load(path, allow_pickle=False) as prior:
            n = len(rows)
            for key, value in identities(rows, protocol["classes"]).items():
                expected = prior[key][:n] if key in ("image_paths", "patient_ids") else prior[key]
                np.testing.assert_array_equal(value, expected)
            for name, old_name in OLD_BRANCHES.items():
                report[f"seed_{seed}/{name}"] = compare(arrays[seed][name], prior[old_name][:n], "historical native " + name)
        register(protocol, [query_path, path, old / f"seed_{seed}" / "support_manifest.json"])
    return report


def alpha_cache(args, protocol, rows, indices, started):
    import torch
    from torch.nn import functional as F
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from eval_query_refinement import CHECKPOINT_SHA, original_helper_check
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator, fit_query_directions
    from view_geometry import apply_view

    seed_everything(1234)
    checkpoint = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("AlphaRAD checkpoint changed")
    register(protocol, [ALPHA / "local_setup" / p for p in ("query_refinement.py", "eval_query_refinement.py")]
             + [PHASE6 / "view_geometry.py", ALPHA / "cxrclip/model/cxrclip_plus_dqn.py",
                ALPHA / "cxrclip/evaluator.py", ALPHA / "cxrclip/base_evaluator.py"])
    for name, current in (("pool", protocol["pool"]), ("test", rows)):
        # Image loader receives no labels.
        with (args.output / f"{name}_inputs.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["image"]); w.writerows([[r["image_path"]] for r in current])
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(checkpoint)]},
              "data_test": {name: {"name": name, "data_type": "image_classification",
                  "data_path": str(args.output / f"{name}_inputs.csv"), "n_class": 3, "normalize": None}
                  for name in ("pool", "test")}, "dataloader": {"test": options}}
    write(args.output / "eval_config.json", config)
    progress(args, "loading_model", started)
    evaluator = Evaluator(config, [str(checkpoint)], "test")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, state["model"])
    del state
    evaluator.model.eval().requires_grad_(False)
    classes = [name.lower() for name in protocol["classes"]]
    positive, prompts = original_queries_from_evaluator(evaluator, classes)
    prompt_path = ALPHA / "local_setup/chexchonet_20261023/prompts.json"
    expected_prompts = json.loads(prompt_path.read_text())
    if prompts != [expected_prompts["tasks"][task]["prompt"] for task in TASKS]:
        raise ValueError("AlphaRAD fixed task prompts changed")
    register(protocol, [prompt_path])
    negative_prompts = [f"There is no {name}." for name in classes]
    negative = np.asarray(evaluator.encode_text(negative_prompts,
        text_max_length=evaluator.config["base"]["dqn_label_max_length"])["cls_emb_raw"], dtype=np.float32)
    matcher = QueryRefinementMatcher(evaluator.model, positive[None], (0.,), use_all_image_tokens=evaluator.use_all_image_tokens)

    def global_features(vision):
        patches = vision["image_patch_raw_features"]
        if len(patches) != 1:
            raise ValueError("Expected one native projected patch layer")
        return F.normalize(patches[0].mean(1), dim=-1)

    parts = {k: [] for k in ("features", "class_local", "logits")}; observed = []
    progress(args, "encoding_support_pool", started, total=len(protocol["pool"]))
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["pool"]:
            vision = matcher.encode_vision(batch["images"])
            native = matcher.match_prepared(vision, positive, return_features=True)
            for k, value in (("features", global_features(vision)), ("class_local", native["class_local"]), ("logits", native["logits"])):
                parts[k].append(value.cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [r["image_path"] for r in protocol["pool"]]:
        raise ValueError("Support pool order changed")
    pool = {k: np.concatenate(v) for k, v in parts.items()}
    queries = {}
    for seed in SEEDS:
        fit_queries = []
        for j, task in enumerate(TASKS):
            ids = indices[seed][task]
            fit = fit_query_directions(pool["class_local"][ids, j:j + 1],
                [[r[task]] for r in protocol["supports"][seed][task]], positive[j:j + 1], etas=(0., .3))
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz", **fit, support_indices=ids)
            fit_queries.append(fit["queries"][1])
        queries[seed] = np.concatenate(fit_queries)
    arrays = {s: {k: [] for k in BRANCHES} for s in SEEDS}
    test = {k: [] for k in ("features_identity", "features_center", "logits_identity", "logits_center")}
    parity = {}; offset = 0
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test"]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != [r["image_path"] for r in rows[offset:stop]]:
                raise ValueError("Test order changed")
            identity = matcher.encode_vision(batch["images"])
            center = matcher.encode_vision(apply_view(batch["images"].to(matcher.device, dtype=torch.float32), "center80"))
            base = matcher.match_prepared(identity, positive)["logits"]
            center_base = matcher.match_prepared(center, positive)["logits"]
            for k, v in (("features_identity", global_features(identity)), ("features_center", global_features(center)),
                         ("logits_identity", base), ("logits_center", center_base)):
                test[k].append(v.cpu().numpy())
            if offset == 0:
                parity["native_author_helpers"] = original_helper_check(evaluator, matcher, batch, classes)
            for seed in SEEDS:
                q = matcher.match_prepared(identity, queries[seed])["logits"]
                qc = matcher.match_prepared(center, queries[seed])["logits"]
                values = dict(original=base, query=q, original_view=(base + center_base) * .5, query_view=(q + qc) * .5)
                for name, value in values.items():
                    arrays[seed][name].append(value.cpu().numpy())
            if offset == 0 or offset // args.batch_size % 25 == 0 or stop == len(rows):
                progress(args, "encoding_test", started, stop, len(rows))
            offset = stop
    for seed in SEEDS:
        logits = {k: np.concatenate(v) for k, v in arrays[seed].items()}
        np.savez_compressed(args.output / f"seed_{seed}" / "native_logits.npz", **identities(rows, protocol["classes"]), **logits)
        arrays[seed] = {k: torch.sigmoid(torch.from_numpy(v)).numpy() for k, v in logits.items()}
    text = dict(positive_raw=positive, negative_raw=negative, positive=positive / np.linalg.norm(positive, axis=-1, keepdims=True),
                negative=negative / np.linalg.norm(negative, axis=-1, keepdims=True), positive_prompts=np.asarray(prompts), negative_prompts=np.asarray(negative_prompts))
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Unfrozen AlphaRAD parameter")
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored,
        feature_definition="L2-normalized mean of frozen native projected patches, class independent",
        native_support_definition="Original-query FLaS class-local contexts; only task-specific ten supports fit Q",
        native_score_domain="sigmoid probability; mean native logits then sigmoid for two views",
        cached_logit_definition="native FLaS binary-presence logit; negative text is comparator-only")
    return pool, {k: np.concatenate(v) for k, v in test.items()}, text, queries, arrays, parity, metadata


def biomed_cache(args, protocol, rows, indices, started):
    import torch
    sys.path.insert(0, str(ROOT / "baselines/local_setup"))
    from evaluate_baseline_buffered import load_adapter
    from evaluate_query_transfer import load_query, check_native
    from query_refinement_common import fit_directions, refine_query, center_pixels
    model_root = ROOT / "baselines/protocols/alpharad_binary/BiomedCLIP"
    register(protocol, [model_root / "local_setup/adapter.py", ROOT / "baselines/BiomedCLIP/local_setup/adapter.py",
        ROOT / "baselines/BiomedCLIP/local_setup/query_refinement_adapter.py", ROOT / "baselines/local_setup/query_refinement_common.py",
        ROOT / "baselines/local_setup/evaluate_baseline_buffered.py", ROOT / "baselines/local_setup/evaluate_query_transfer.py"])
    manifest_path = ROOT / "baselines/BiomedCLIP/local_setup/manifest.json"
    manifest = json.loads(manifest_path.read_text()); checkpoint = manifest["checkpoints"][0]
    if sha(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("BiomedCLIP checkpoint changed")
    register(protocol, [manifest_path])
    progress(args, "loading_model", started)
    native = load_adapter(model_root, "cuda"); query = load_query("BiomedCLIP", native)
    classes = protocol["classes"]; parts = []; logit_parts = []
    def raw_score(features, original):
        values = features @ original.T
        return values[:, 3:] - values[:, :3]
    with torch.inference_mode():
        original = query._queries(classes)
        paths = [r["image_path"] for r in protocol["pool"]]
        progress(args, "encoding_support_pool", started, total=len(paths))
        for offset in range(0, len(paths), args.batch_size):
            x = query._images(paths[offset:offset + args.batch_size], "identity")
            z = native.model.encode_image(x, normalize=True).float()
            parts.append(z.cpu()); logit_parts.append(raw_score(z, original).cpu().numpy())
    features = torch.cat(parts); pool = dict(features=features.numpy(), logits=np.concatenate(logit_parts))
    positive = original[3:].float().cpu(); queries = {}
    for seed in SEEDS:
        fits = []
        for j, task in enumerate(TASKS):
            ids = indices[seed][task]
            fit = fit_directions(features[ids], [[r[task]] for r in protocol["supports"][seed][task]])
            revised = refine_query(positive[j:j + 1], fit["directions"], eta=.3, preserve_norm=True)
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz",
                **{k: v.numpy() for k, v in fit.items()}, original_queries=positive[j:j + 1].numpy(), queries=revised.numpy(), support_indices=ids)
            fits.append(revised)
        queries[seed] = torch.cat(fits).numpy()
    gpu_queries = {s: torch.as_tensor(q, device=native.device) for s, q in queries.items()}
    test = {k: [] for k in ("features_identity", "features_center", "logits_identity", "logits_center")}
    arrays = {s: {k: [] for k in BRANCHES} for s in SEEDS}; parity = {}
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]; paths = [r["image_path"] for r in batch]
            x = query._images(paths, "identity")
            z = native.model.encode_image(x, normalize=True).float()
            zc = native.model.encode_image(center_pixels(x, "center80"), normalize=True).float()
            base = query._score(z, original, 3); bc = query._score(zc, original, 3)
            for k, v in (("features_identity", z), ("features_center", zc),
                         ("logits_identity", raw_score(z, original)), ("logits_center", raw_score(zc, original))):
                test[k].append(v.cpu().numpy())
            if offset == 0:
                parity["native_adapter"] = check_native(base, native.predict(paths, classes))
            for seed in SEEDS:
                q = query._score(z, original, 3, gpu_queries[seed]); qc = query._score(zc, original, 3, gpu_queries[seed])
                for k, v in dict(original=base, query=q, original_view=(base + bc) * .5, query_view=(q + qc) * .5).items():
                    arrays[seed][k].append(v)
            done = offset + len(batch)
            if offset == 0 or offset // args.batch_size % 25 == 0 or done == len(rows):
                progress(args, "encoding_test", started, done, len(rows))
    phrases = [name.replace("_", " ") for name in classes]
    p = original[3:].float().cpu().numpy(); n = original[:3].float().cpu().numpy()
    text = dict(positive=p, negative=n, positive_raw=p, negative_raw=n,
        positive_prompts=np.asarray([f"There is {name}." for name in phrases]),
        negative_prompts=np.asarray([f"There is no {name}." for name in phrases]))
    if any(p.requires_grad for p in native.model.parameters()):
        raise ValueError("Unfrozen BiomedCLIP parameter")
    metadata = dict(checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"],
        feature_definition="L2-normalized native projected global image embedding",
        native_support_definition="same global embedding; each task fits only its ten supports",
        native_score_domain="temperature-one binary-softmax probability; two views average probabilities",
        cached_logit_definition="positive cosine minus negative cosine, temperature one", native_metadata=native.metadata)
    return pool, {k: np.concatenate(v) for k, v in test.items()}, text, queries, {s: {k: np.concatenate(v) for k,v in a.items()} for s,a in arrays.items()}, parity, metadata


def run(args):
    import torch
    started = time.time(); args.output.mkdir(parents=True, exist_ok=False)
    protocol, _ = load_protocol_without_test_labels(args.protocol)
    register(protocol, [Path(__file__), STABILITY / "infer_multiseed.py"])
    rows = protocol["test"][:args.smoke] if args.smoke else protocol["test"]
    indices = support_indices(protocol, args.output)
    torch.set_num_threads(4); torch.manual_seed(1234); np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    fn = alpha_cache if args.model == "AlphaRAD" else biomed_cache
    pool, test, text, queries, arrays, parity, metadata = fn(args, protocol, rows, indices, started)
    for seed in SEEDS:
        for name, values in arrays[seed].items():
            if values.shape != (len(rows),3) or not np.isfinite(values).all():
                raise ValueError("Malformed native predictions")
        for name in ("original", "original_view"):
            np.testing.assert_array_equal(arrays[seed][name], arrays[SEEDS[0]][name])
    for name, value in test.items():
        if value.shape[0] != len(rows) or not np.isfinite(value).all():
            raise ValueError("Malformed test features")
    parity["historical_five_seed_endpoints"] = historical_checks(args, protocol, rows, arrays,
        pool["class_local"] if args.model == "AlphaRAD" else pool["features"], queries)
    np.savez_compressed(args.output / "test_features.npz", **identities(rows, protocol["classes"]), **test)
    np.savez_compressed(args.output / "support_pool.npz", **identities(protocol["pool"], protocol["classes"]),
        **pool)
    np.savez_compressed(args.output / "queries.npz", class_names=np.asarray(protocol["classes"]), task_keys=np.asarray(TASKS), **text)
    for seed in SEEDS:
        np.savez_compressed(args.output / f"seed_{seed}" / "native_predictions.npz",
            **identities(rows, protocol["classes"]), seed=np.asarray(seed), **arrays[seed])
        np.save(args.output / f"seed_{seed}" / "queries.npy", queries[seed], allow_pickle=False)
    # Test labels remain solely in the protocol's separate scoring asset. This
    # worker never opens, copies, or exports that asset, even after inference.
    changed = [p for p, digest in protocol["sources"].items() if sha(p) != digest]
    if changed:
        raise RuntimeError("Protocol/source changed during caching: " + repr(changed))
    write(args.output / "source_sha256.json", protocol["sources"])
    write(args.output / "native_parity.json", parity)
    status = "smoke_complete" if args.smoke else "complete"
    artifacts = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*.npz")}
    manifest = dict(status=status, model=args.model,
        protocol=str(protocol["path"]), protocol_sha256=sha(protocol["path"]),
        test_images=len(rows), full_test_images=7407, full_test_patients=2470,
        support_pool_images=len(protocol["pool"]), task_keys=list(TASKS), class_names=protocol["classes"],
        seeds=list(SEEDS), eta=.3, center_crop_ratio=.8, views=["identity","center80"],
        feature_dimension=int(pool["features"].shape[1]), metadata=metadata,
        feature_scope="competition uses class-independent global features; native AlphaRAD Q uses class-local features",
        fit_visible_test_files_have_no_labels=True, test_labels_loaded_or_exported=False,
        comparator_support_view="identity only", comparator_fit_reused_across_test_views=True,
        test_metrics_computed=False, no_gradient_training=True,
        reuse="Frozen weights, preprocessing, prompts, data/splits reused; newly encoded global view features and native predictions numerically audited against all five completed stability runs",
        artifact_sha256=artifacts, elapsed_seconds=time.time()-started,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_gpu=args.gpu,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(args.output / "CACHE_MANIFEST.json", manifest)
    write(args.output / "result.json", dict(status=status, model=args.model, rows=len(rows),
        feature_dimension=manifest["feature_dimension"], cache_manifest_sha256=sha(args.output / "CACHE_MANIFEST.json"),
        protocol_sha256=manifest["protocol_sha256"], artifact_sha256=artifacts,
        native_parity=parity, test_metrics_computed=False, elapsed_seconds=time.time()-started))
    progress(args, status, started, len(rows), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("BiomedCLIP", "AlphaRAD"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--gpu", type=int, choices=(6, 7), help="Physical GPU provenance only; parent controls CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error("Invalid batch, workers or smoke value")
    args.output = args.output.resolve()
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir():
            write(args.output / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
