#!/usr/bin/env python3
"""Five support draws on one frozen CheXchoNet holdout, sharing image forwards.

This is a sampling/stability experiment, not model selection. Each task/seed
fits only its own ten supports; the other candidate labels never enter its fit.
The historical native probability definitions and center80 fusion are retained.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

MEDICAL = Path(__file__).resolve().parents[2]
ALPHA = MEDICAL / "AlphaRAD"
BASELINES = MEDICAL / "baselines"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
OLD_ALPHA = MEDICAL / "experiments/chexchonet_comparison_20261023_233926/alpharad_paired"
OLD_BIO = BASELINES / "runs/chexchonet_20261024_000447/BiomedCLIP_full"
POOL = ALPHA / "local_setup/chexchonet_20261023/excluded_patients_images.csv"
TASKS = ("slvh", "dlv", "composite_slvh_dlv")
DATASETS = tuple("chexchonet_" + task for task in TASKS)
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    import numpy as np
    def default(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=default) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=None)
def canonical(path):
    return str(Path(path).resolve())


@lru_cache(maxsize=None)
def read_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        image = canonical(row.get("image_path", row.get("image", "")))
        pid = str(row["patient_id"]).strip()
        labels = {task: int(float(row[task])) for task in TASKS}
        if not pid or not Path(image).is_file() or any(y not in (0, 1) for y in labels.values()):
            raise ValueError("Invalid labelled patient/image record")
        if labels[TASKS[2]] != int(labels[TASKS[0]] or labels[TASKS[1]]):
            raise ValueError("Composite label is not CSLV OR CDLV")
        out.append(dict(image_path=image, patient_id=pid, **labels))
    if not out or len({r["image_path"] for r in out}) != len(out):
        raise ValueError("Empty or duplicate input manifest")
    return out


def signature(rows):
    return [(r["image_path"], r["patient_id"], *(r[t] for t in TASKS)) for r in rows]


def load_protocol(path):
    """Validate both adapters' manifests, fixed cohort and task-only supports."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "catalog.json"
    master = json.loads(path.read_text())
    definition = master["protocol"]
    if (master.get("status") != "complete" or definition["support_seeds"] != list(SEEDS)
            or definition["eta"] != .3 or definition["center_crop_ratio"] != .8
            or definition["test_image_count"] != 7407 or definition["test_patient_count"] != 2470):
        raise ValueError("Protocol is not the frozen five-seed stability experiment")
    sources = {str(path): sha(path), str(POOL): sha(POOL)}
    pool = read_rows(POOL)
    if len(pool) != 87 or len({r["patient_id"] for r in pool}) != 30:
        raise ValueError("Historical outside-test candidate pool must remain 87 images / 30 patients")
    pool_by_path = {r["image_path"]: r for r in pool}
    supports, test, classes = {}, None, None
    for seed in SEEDS:
        seed_entry = master["seeds"][str(seed)]
        alpha_path, bio_path = [Path(seed_entry[k]) for k in ("alpharad_catalog", "baseline_catalog")]
        for child, expected in ((alpha_path, seed_entry["alpharad_catalog_sha256"]),
                                (bio_path, seed_entry["baseline_catalog_sha256"])):
            if sha(child) != expected:
                raise ValueError("Seed catalog differs from frozen protocol: " + str(child))
        sources.update({str(p): sha(p) for p in (alpha_path, bio_path)})
        alpha_catalog, bio_catalog = [json.loads(p.read_text()) for p in (alpha_path, bio_path)]
        supports[seed] = {}
        names = []
        for task, dataset in zip(TASKS, DATASETS):
            entry = alpha_catalog["datasets"][dataset]
            for field, hashfield in (("csv_path", "csv_sha256"),
                                     ("support_csv_path", "support_csv_sha256")):
                actual = sha(entry[field])
                if actual != entry[hashfield]:
                    raise ValueError("AlphaRAD manifest changed: " + entry[field])
                sources[entry[field]] = actual
            selected, current = read_rows(entry["support_csv_path"]), read_rows(entry["csv_path"])
            if test is None:
                test = current
            elif signature(test) != signature(current):
                raise ValueError("All seeds/tasks must retain identical ordered test images, patients and labels")
            if len(current) != 7407 or len({r["patient_id"] for r in current}) != 2470:
                raise ValueError("Fixed holdout must remain 7407 images / 2470 patients")
            if len(selected) != 10 or sum(r[task] for r in selected) != 5 or len({r["patient_id"] for r in selected}) != 10:
                raise ValueError("Each seed/task requires five positive and five negative distinct support patients")
            if {r["patient_id"] for r in selected} & {r["patient_id"] for r in current}:
                raise ValueError("Support/test patient overlap")
            for row in selected:
                if row != pool_by_path.get(row["image_path"]):
                    raise ValueError("Support row lies outside the fixed historical candidate pool")
                if row[task] == 0 and any(r[task] for r in pool if r["patient_id"] == row["patient_id"]):
                    raise ValueError("Negative support patient has another positive candidate image")
            be = bio_catalog["datasets"][dataset]
            for field, hashfield in (("manifest", "sha256"), ("support_manifest", "support_manifest_sha256")):
                actual = sha(be[field])
                if actual != be[hashfield]:
                    raise ValueError("BiomedCLIP manifest changed: " + be[field])
                sources[be[field]] = actual
            bd, bs = [json.loads(Path(be[k]).read_text()) for k in ("manifest", "support_manifest")]
            if bd["class_names"] != entry["class_names"] or len(entry["class_names"]) != 1:
                raise ValueError("Cross-model class names differ")
            def binary_signature(items):
                return [(canonical(r["image"]), str(r["patient_id"]), int(r["labels"][0])) for r in items]
            if binary_signature(bd["rows"]) != [(r["image_path"], r["patient_id"], r[task]) for r in current]:
                raise ValueError("Cross-model evaluation rows/labels differ")
            if binary_signature(bs["support_rows"]) != [(r["image_path"], r["patient_id"], r[task]) for r in selected]:
                raise ValueError("Cross-model support rows/labels differ")
            for row in bs["support_rows"]:
                if "byte_sha256" in row and sha(row["image"]) != row["byte_sha256"]:
                    raise ValueError("Support image bytes changed")
                sources[str(Path(row["image"]).resolve())] = sha(row["image"])
            names.append(entry["class_names"][0])
            supports[seed][task] = selected
        if classes is None:
            classes = names
        elif classes != names:
            raise ValueError("Seed class names/order differ")
    if {r["patient_id"] for r in pool} & {r["patient_id"] for r in test}:
        raise ValueError("Candidate-pool patient leaked into fixed test set")
    return dict(master=master, path=path, sources=sources, pool=pool, test=test,
                supports=supports, classes=classes)


def progress(args, stage, started, done=0, total=0):
    record = dict(status=stage, model=args.model, done=done, total=total,
                  elapsed_seconds=time.time() - started, seeds=list(SEEDS), smoke=args.smoke)
    write(args.output / "status.json", record)
    print(json.dumps(record), flush=True)


def historical_parity(model, rows, arrays):
    """Only unchanged arms are checked; no metric-driven choices are made."""
    import numpy as np
    report = {}
    if model == "AlphaRAD":
        old_paths = json.loads((OLD_ALPHA / "image_paths.json").read_text())
        indices = {str(Path(p).resolve()): i for i, p in enumerate(old_paths)}
        ids = [indices[r["image_path"]] for r in rows]
        expected_labels = np.load(OLD_ALPHA / "labels.npy", allow_pickle=False)[ids]
        np.testing.assert_array_equal(expected_labels, [[r[t] for t in TASKS] for r in rows])
        for branch in ("original", "tta_center_original"):
            expected = np.load(OLD_ALPHA / f"{branch}_probabilities.npy", allow_pickle=False)[ids]
            actual = arrays[SEEDS[0]][branch]
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
            report[branch] = dict(passed=True, reference=str(OLD_ALPHA / f"{branch}_probabilities.npy"),
                reference_sha256=sha(OLD_ALPHA / f"{branch}_probabilities.npy"),
                max_abs_error=float(np.max(np.abs(actual - expected))), bitwise_equal=bool(np.array_equal(actual, expected)))
    else:
        for j, dataset in enumerate(DATASETS):
            prediction = OLD_BIO / dataset / "predictions.npz"
            with np.load(prediction, allow_pickle=False) as old:
                indices = {str(Path(p).resolve()): i for i, p in enumerate(old["image_paths"].tolist())}
                ids = [indices[r["image_path"]] for r in rows]
                np.testing.assert_array_equal(old["labels"][ids, 0], [r[TASKS[j]] for r in rows])
                for branch in ("original", "tta_center_original"):
                    actual, expected = arrays[SEEDS[0]][branch][:, j], old[branch][ids, 0]
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
                    report[dataset + "/" + branch] = dict(passed=True, reference=str(prediction),
                        reference_sha256=sha(prediction), max_abs_error=float(np.max(np.abs(actual - expected))),
                        bitwise_equal=bool(np.array_equal(actual, expected)))
    return dict(passed=True, images=len(rows), atol=2e-6, rtol=1e-5, checks=report)


def alpha_inference(args, protocol, rows, started):
    import numpy as np
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from eval_query_refinement import CHECKPOINT_SHA, original_helper_check
    from eval_chexchonet import native_manifest
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator, fit_query_directions
    from view_geometry import apply_view, aggregate_outputs

    seed_everything(1234)
    checkpoint_path = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
    if sha(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint changed")
    source_paths = [ALPHA / "local_setup" / name for name in
        ("query_refinement.py", "eval_query_refinement.py", "eval_chexchonet.py")]
    source_paths += [PHASE6 / "view_geometry.py", ALPHA / "cxrclip/model/cxrclip_plus_dqn.py",
                     ALPHA / "cxrclip/evaluator.py", ALPHA / "cxrclip/base_evaluator.py"]
    protocol["sources"].update({str(p): sha(p) for p in source_paths})
    native_manifest(args.output / "pool_inputs.csv", protocol["pool"], TASKS)
    native_manifest(args.output / "test_inputs.csv", rows, TASKS)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(checkpoint_path)]},
        "data_test": {name: {"name": name, "data_type": "image_classification",
            "data_path": str(args.output / f"{name}_inputs.csv"), "n_class": 3,
            "normalize": None} for name in ("pool", "test")}, "dataloader": {"test": options}}
    write(args.output / "eval_config.json", config)
    progress(args, "loading", started)
    evaluator = Evaluator(config, [str(checkpoint_path)], "test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    classes = [name.lower() for name in protocol["classes"]]
    original, prompts = original_queries_from_evaluator(evaluator, classes)
    prompt_path = ALPHA / "local_setup/chexchonet_20261023/prompts.json"
    prompt_record = json.loads(prompt_path.read_text())
    if prompts != [prompt_record["tasks"][task]["prompt"] for task in TASKS]:
        raise ValueError("Frozen CheXchoNet prompts changed")
    protocol["sources"][str(prompt_path)] = sha(prompt_path)
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    parts, observed = [], []
    progress(args, "encoding_support_pool", started, total=len(protocol["pool"]))
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["pool"]:
            vision = matcher.encode_vision(batch["images"])
            parts.append(matcher.match_prepared(vision, original, return_features=True)["class_local"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [r["image_path"] for r in protocol["pool"]]:
        raise ValueError("Candidate pool order changed")
    features = np.concatenate(parts)
    np.save(args.output / "candidate_class_local.npy", features, allow_pickle=False)
    lookup = {r["image_path"]: i for i, r in enumerate(protocol["pool"])}
    queries = {}
    for seed in SEEDS:
        fits = []
        for j, task in enumerate(TASKS):
            supports = protocol["supports"][seed][task]
            ids = [lookup[r["image_path"]] for r in supports]
            fit = fit_query_directions(features[ids, j:j + 1], [[r[task]] for r in supports],
                                       original[j:j + 1], etas=(0., .3))
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz",
                                **fit, support_indices=np.asarray(ids))
            fits.append(fit["queries"][1])
        queries[seed] = np.concatenate(fits)
        np.save(args.output / f"seed_{seed}" / "queries.npy", queries[seed], allow_pickle=False)
    arrays = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    logits_out = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    offset, parity = 0, {}
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test"]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != [r["image_path"] for r in rows[offset:stop]]:
                raise ValueError("Test image order changed")
            identity = matcher.encode_vision(batch["images"])
            center = matcher.encode_vision(apply_view(batch["images"].to(matcher.device, dtype=torch.float32), "center80"))
            original_outputs = {"identity": matcher.match_prepared(identity, original),
                                "center80": matcher.match_prepared(center, original)}
            base = original_outputs["identity"]["logits"]
            view = aggregate_outputs(original_outputs, ["identity", "center80"])["logits"]
            if offset == 0:
                parity["author_helpers"] = original_helper_check(evaluator, matcher, batch, classes)
            for seed in SEEDS:
                outputs = {"identity": matcher.match_prepared(identity, queries[seed]),
                           "center80": matcher.match_prepared(center, queries[seed])}
                values = dict(original=base, tta_center_original=view,
                    query_refined=outputs["identity"]["logits"],
                    tta_center_refined=aggregate_outputs(outputs, ["identity", "center80"])["logits"])
                for branch, logits in values.items():
                    logits_out[seed][branch].append(logits.cpu().numpy())
            if offset == 0 or offset // args.batch_size % 25 == 0 or stop == len(rows):
                progress(args, "inference", started, stop, len(rows))
            offset = stop
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("AlphaRAD parameters must remain frozen")
    for seed in SEEDS:
        joined_logits = {key: np.concatenate(value) for key, value in logits_out[seed].items()}
        # Historical eval_chexchonet applies sigmoid on the full CPU logit
        # matrix. Retain that operation order, including floating-point ties.
        arrays[seed] = {key: [torch.sigmoid(torch.from_numpy(value)).numpy()]
                        for key, value in joined_logits.items()}
        np.savez_compressed(args.output / f"seed_{seed}" / "logits.npz",
                            **joined_logits)
    return arrays, parity, dict(checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored,
        prompts=prompts, classification_aggregation="mean native logits then sigmoid",
        support_feature="native original-query class-local FLaS context", no_gradient_training=True)


def biomed_inference(args, protocol, rows, started):
    import numpy as np
    import torch
    sys.path.insert(0, str(BASELINES / "local_setup"))
    from evaluate_baseline_buffered import load_adapter
    from evaluate_query_transfer import load_query, check_native
    from query_refinement_common import fit_directions, refine_query, center_pixels

    model_root = BASELINES / "protocols/alpharad_binary/BiomedCLIP"
    source_paths = [BASELINES / "local_setup" / name for name in
        ("evaluate_baseline_buffered.py", "evaluate_query_transfer.py", "query_refinement_common.py")]
    source_paths += [model_root / "local_setup/adapter.py", BASELINES / "BiomedCLIP/local_setup/adapter.py",
                     BASELINES / "BiomedCLIP/local_setup/query_refinement_adapter.py"]
    protocol["sources"].update({str(p): sha(p) for p in source_paths})
    progress(args, "loading", started)
    native = load_adapter(model_root, "cuda")
    query = load_query("BiomedCLIP", native)
    classes = protocol["classes"]
    paths = [r["image_path"] for r in protocol["pool"]]
    parts = []
    progress(args, "encoding_support_pool", started, total=len(paths))
    with torch.inference_mode():
        originals = query._queries(classes)
        for offset in range(0, len(paths), args.batch_size):
            pixels = query._images(paths[offset:offset + args.batch_size], "identity")
            parts.append(native.model.encode_image(pixels, normalize=True).float().cpu())
    features = torch.cat(parts)
    np.save(args.output / "candidate_global_features.npy", features.numpy(), allow_pickle=False)
    lookup = {p: i for i, p in enumerate(paths)}
    positive = originals[3:].float().cpu()
    queries = {}
    for seed in SEEDS:
        fitted = []
        for j, task in enumerate(TASKS):
            supports = protocol["supports"][seed][task]
            ids = [lookup[r["image_path"]] for r in supports]
            fit = fit_directions(features[ids], [[r[task]] for r in supports])
            revised = refine_query(positive[j:j + 1], fit["directions"], eta=.3, preserve_norm=True)
            np.savez_compressed(args.output / f"seed_{seed}" / f"query_fit_{task}.npz",
                **{key: value.numpy() for key, value in fit.items()}, original_queries=positive[j:j + 1].numpy(),
                queries=revised.numpy(), original_negative_queries=originals[j:j + 1].float().cpu().numpy(),
                support_indices=np.asarray(ids))
            fitted.append(revised)
        queries[seed] = torch.cat(fitted).to(native.device)
        np.save(args.output / f"seed_{seed}" / "queries.npy", queries[seed].cpu().numpy(), allow_pickle=False)
    arrays = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    parity = {"independent_support_fits": {}}
    # This is a real adapter-equivalence check, not an alternative prediction rule.
    with torch.inference_mode():
        for j, task in enumerate(TASKS):
            supports = protocol["supports"][SEEDS[0]][task]
            individual = load_query("BiomedCLIP", native)
            individual.fit([r["image_path"] for r in supports], [[r[task]] for r in supports],
                           [classes[j]], batch_size=args.batch_size)
            expected, actual = individual._positive.cpu(), queries[SEEDS[0]][j:j + 1].cpu()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
            parity["independent_support_fits"][task] = dict(passed=True,
                max_abs_error=float((actual - expected).abs().max()))
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            image_paths = [r["image_path"] for r in batch]
            pixels = query._images(image_paths, "identity")
            identity = native.model.encode_image(pixels, normalize=True)
            center = native.model.encode_image(center_pixels(pixels, "center80"), normalize=True)
            base = query._score(identity, originals, 3)
            center_base = query._score(center, originals, 3)
            if offset == 0:
                parity["native_adapter"] = check_native(base, native.predict(image_paths, classes))
            for seed in SEEDS:
                refined = query._score(identity, originals, 3, queries[seed])
                center_refined = query._score(center, originals, 3, queries[seed])
                values = dict(original=base, query_refined=refined,
                              tta_center_original=(base + center_base) * .5,
                              tta_center_refined=(refined + center_refined) * .5)
                for branch, value in values.items():
                    arrays[seed][branch].append(value)
            done = offset + len(batch)
            if offset == 0 or offset // args.batch_size % 50 == 0 or done == len(rows):
                progress(args, "inference", started, done, len(rows))
    if any(p.requires_grad for p in native.model.parameters()):
        raise ValueError("BiomedCLIP parameters must remain frozen")
    return arrays, parity, dict(native_metadata=native.metadata,
        checkpoint_sha256=sha(native.metadata["checkpoint"]), classification_aggregation="mean native binary-softmax probabilities",
        negative_queries="unchanged", temperature=1., eta=.3, no_gradient_training=True)


def run(args):
    import numpy as np
    import torch
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(args.protocol)
    protocol["sources"][str(Path(__file__).resolve())] = sha(__file__)
    rows = protocol["test"][:args.smoke] if args.smoke else protocol["test"]
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        write(folder / "support_manifest.json", dict(seed=seed,
            task_support_rows=protocol["supports"][seed], eta=.3, support_images_per_task=10,
            test_labels_used_for_fit=False, original_historical_seed42_is_not_this_seed42=True))
    write(args.output / "input_audit.json", dict(protocol=str(protocol["path"]),
        protocol_sha256=sha(protocol["path"]), candidate_images=87, candidate_patients=30,
        fixed_test_images=7407, fixed_test_patients=2470, inference_images=len(rows),
        seeds=list(SEEDS), task_keys=list(TASKS), class_names=protocol["classes"],
        supports_per_task=10, positive_supports=5, negative_supports=5,
        cross_task_patient_reuse_allowed=True, support_test_patient_overlap=0,
        support_negative_patient_rule="all candidate images for the task are negative"))
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    fn = alpha_inference if args.model == "AlphaRAD" else biomed_inference
    chunks, parity, metadata = fn(args, protocol, rows, started)
    arrays = {seed: {key: np.concatenate(value) for key, value in branches.items()}
              for seed, branches in chunks.items()}
    labels = np.asarray([[r[task] for task in TASKS] for r in rows], dtype=np.uint8)
    for seed, branches in arrays.items():
        for key, values in branches.items():
            if values.shape != (len(rows), 3) or not np.isfinite(values).all():
                raise ValueError("Malformed prediction array")
        for key in ("original", "tta_center_original"):
            np.testing.assert_array_equal(branches[key], arrays[SEEDS[0]][key])
    parity["historical_unchanged_arms"] = historical_parity(args.model, rows, arrays)
    parity["unchanged_arms_bitwise_equal_across_seeds"] = True
    write(args.output / "native_parity.json", parity)
    sys.path.insert(0, str(BASELINES / "local_setup"))
    from evaluate_baseline_buffered import classification_metrics
    results = {}
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        np.savez_compressed(folder / "predictions.npz", labels=labels,
            image_paths=np.asarray([r["image_path"] for r in rows]),
            patient_ids=np.asarray([r["patient_id"] for r in rows]),
            class_names=np.asarray(protocol["classes"]), task_keys=np.asarray(TASKS),
            seed=np.asarray(seed), **arrays[seed])
        metrics = {key: classification_metrics(labels, values, protocol["classes"])
                   for key, values in arrays[seed].items()}
        result = dict(status="smoke_complete" if args.smoke else "complete", model=args.model,
            seed=seed, rows=len(rows), patients=len(set(r["patient_id"] for r in rows)),
            branches=metrics, eta=.3, views=["identity", "center80"], metadata=metadata,
            same_fixed_test_all_seeds=True, no_test_selection=True,
            prediction_sha256=sha(folder / "predictions.npz"))
        write(folder / "result.json", result)
        results[str(seed)] = result
    changed = [p for p, digest in protocol["sources"].items() if sha(p) != digest]
    if changed:
        raise ValueError("Protocol/source changed during inference: " + repr(changed))
    write(args.output / "source_sha256.json", protocol["sources"])
    status = "smoke_complete" if args.smoke else "complete"
    write(args.output / "result.json", dict(status=status, model=args.model, seeds=list(SEEDS),
        rows=len(rows), seed_results=results, native_parity=parity, metadata=metadata,
        batch_size=args.batch_size, elapsed_seconds=time.time() - started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_version=torch.__version__))
    progress(args, status, started, len(rows), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("AlphaRAD", "BiomedCLIP"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0, help="Only infer first N fixed-test images; still fit all five supports")
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
