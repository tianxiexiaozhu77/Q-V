#!/usr/bin/env python3
"""Fresh single-view AlphaRAD query-path inference on sealed SIIM header groups.

Only support labels and a label-free test identity manifest are read. Test RLE,
masks and classification labels stay in a separate evaluation manifest, which
this worker never opens. Run under the controller's GPU lease, not directly on
an occupied GPU. Four pathways are fixed before inference; no branch is chosen
from outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MEDICAL = HERE.parents[1]
ALPHA = MEDICAL / "AlphaRAD"
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = ("original", "readout_only", "aggregation_only", "query")
FORBIDDEN_TEST_KEYS = {"label", "labels", "rle", "rles", "encodedpixels", "encoded_pixels",
                       "mask", "masks", "mask_path", "segmentation"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def checked(path, digest):
    path = Path(path).resolve()
    if not digest or sha(path) != digest:
        raise ValueError("Frozen file checksum changed: " + str(path))
    return path


def resolve_spec(spec, root):
    if not isinstance(spec, dict) or not {"path", "sha256"} <= set(spec):
        raise ValueError("A manifest must specify path and SHA256")
    path = Path(spec["path"])
    return checked(path if path.is_absolute() else root / path, spec["sha256"])


def manifest_rows(value):
    if isinstance(value, list):
        return value
    for key in ("rows", "test_rows", "test_identities"):
        if key in value:
            return value[key]
    raise ValueError("Test identity manifest contains no rows")


def normalize_row(row, *, support):
    if not support and {str(key).lower() for key in row} & FORBIDDEN_TEST_KEYS:
        raise ValueError("Test inference manifest must contain no labels, RLE or masks")
    image = checked(row["image"], row["byte_sha256"])
    group = str(row.get("group_id", ""))
    if not group:
        raise ValueError("Every image needs a nonempty frozen DICOM header group identity")
    result = {"image": str(image), "group_id": group, "byte_sha256": row["byte_sha256"]}
    if row.get("pixel_sha256"):
        # Pixel fingerprints were frozen by protocol preparation. The worker
        # authenticates PNG bytes and uses these declarations to reject the
        # same decoded image saved with different PNG compression.
        result["pixel_sha256"] = row["pixel_sha256"]
    if support:
        if row.get("labels") not in ([0], [1]):
            raise ValueError("SIIM support needs one binary label")
        result["labels"] = list(row["labels"])
    return result


def load_protocol(path, smoke=0):
    """Authenticate identities and support labels, without opening evaluation."""
    path = Path(path).resolve()
    catalog = read(path)
    if (catalog.get("dataset") != "siim_new_holdout" or
            catalog.get("class_names") != ["Pneumothorax"] or float(catalog.get("eta", -1)) != .3):
        raise ValueError("Unexpected SIIM dataset, classes or fixed eta")
    if set(catalog["seeds"]) != {str(s) for s in SEEDS}:
        raise ValueError("Exactly the five fixed support seeds are required")
    identity_path = resolve_spec(catalog["test_identity_manifest"], path.parent)
    raw_test = manifest_rows(read(identity_path))
    test = [normalize_row(row, support=False) for row in raw_test]
    if not test or len({r["image"] for r in test}) != len(test):
        raise ValueError("Test identity list is empty or repeats images")
    if "test_count" in catalog and catalog["test_count"] != len(test):
        raise ValueError("Declared test image count changed")
    test_groups = {r["group_id"] for r in test}
    test_hashes = {r["byte_sha256"] for r in test}
    test_pixels = {r["pixel_sha256"] for r in test if r.get("pixel_sha256")}
    sources = {str(path): sha(path), str(identity_path): sha(identity_path)}
    supports, union = {}, {}
    for seed in SEEDS:
        rows = [normalize_row(row, support=True) for row in catalog["seeds"][str(seed)]["support_rows"]]
        if (len(rows) != 10 or sum(r["labels"][0] for r in rows) != 5 or
                len({r["group_id"] for r in rows}) != 10 or len({r["image"] for r in rows}) != 10):
            raise ValueError("Each seed requires five positive and five negative distinct DICOM header groups")
        if ({r["group_id"] for r in rows} & test_groups or
                {r["byte_sha256"] for r in rows} & test_hashes or
                {r["pixel_sha256"] for r in rows if r.get("pixel_sha256")} & test_pixels):
            raise ValueError("Support and test share a DICOM header group or exact image content")
        for row in rows:
            if row["image"] in union and union[row["image"]] != row:
                raise ValueError("Cross-seed support identity or labels disagree")
            union[row["image"]] = row
        supports[seed] = rows
    union = [union[name] for name in sorted(union)]
    for row in test + union:
        sources[row["image"]] = row["byte_sha256"]
    # evaluation_manifest is deliberately not resolved, hashed, or opened.
    return {"catalog": catalog, "classes": ["Pneumothorax"], "test_all": test,
            "test": test[:smoke] if smoke else test, "supports": supports,
            "union": union, "sources": sources, "identity_path": identity_path}


def source_files():
    return ([Path(__file__).resolve(), HERE / "intervention.py", CHECKPOINT,
             ALPHA / "local_setup/query_refinement.py", ALPHA / "local_setup/python_local.sh"]
            + sorted((ALPHA / "cxrclip").rglob("*.py")))


def write_images(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        writer.writerows([[row["image"]] for row in rows])


def run(args):
    import numpy as np
    import torch
    from PIL import Image
    sys.path[:0] = [str(HERE), str(ALPHA), str(ALPHA / "local_setup")]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import (QueryRefinementMatcher, original_queries_from_evaluator,
                                  fit_query_directions)
    from intervention import evaluate_interventions, audit_native_equivalence

    started = time.time()

    def progress(status, **extra):
        value = dict(status=status, model="AlphaRAD", dataset="siim_new_holdout",
                     smoke=args.smoke, elapsed_seconds=time.time()-started, **extra)
        write_json(args.output / "status.json", value)
        print(json.dumps(value), flush=True)

    progress("authenticating_inputs")
    job = load_protocol(args.protocol, args.smoke)
    sources = {**job["sources"], **{str(path): sha(path) for path in source_files()}}
    if sources[str(CHECKPOINT)] != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint changed")
    geometry = {}
    for row in job["union"] + job["test"]:
        with Image.open(row["image"]) as image:
            if image.size != (512, 512):
                raise ValueError("SIIM author conversion must be a 512x512 square: " + row["image"])
            geometry[row["image"]] = list(image.size)
    write_json(args.output / "source_sha256.json", sources)
    write_json(args.output / "support_union.json", job["union"])
    write_json(args.output / "input_geometry.json", geometry)
    for seed in SEEDS:
        (args.output / f"seed_{seed}").mkdir()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    for name, rows in (("support", job["union"]), ("test", job["test"])):
        write_images(args.output / f"{name}_inputs.csv", rows)
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(CHECKPOINT)]},
        "data_test": {name: dict(name=name, data_type="image_classification",
            data_path=str(args.output / f"{name}_inputs.csv"), n_class=1, normalize=None)
            for name in ("support", "test")}, "dataloader": {"test": options}}
    write_json(args.output / "eval_config.json", config)
    progress("loading_model")
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    native_transform = evaluator.config["transform"]["test"]
    if native_transform != {"Resize": {"size": 518}, "CenterCrop": {"size": 518}}:
        raise ValueError("Native test geometry changed: " + repr(native_transform))
    original, prompts = original_queries_from_evaluator(evaluator, job["classes"])
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    original_tensor = torch.as_tensor(original, device=matcher.device)
    support_parts = {"class_local": [], "logits": []}
    observed = []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support"]:
            if tuple(batch["images"].shape[-2:]) != (518, 518):
                raise ValueError("Native support image input is not 518x518")
            vision = matcher.encode_vision(batch["images"])
            result = matcher.match_prepared(vision, original_tensor, return_features=True)
            for key in support_parts:
                support_parts[key].append(result[key].cpu().numpy())
            observed.extend(batch["image_paths"])
    if observed != [r["image"] for r in job["union"]]:
        raise ValueError("Support loader reordered images")
    support = {key: np.concatenate(parts) for key, parts in support_parts.items()}
    np.savez_compressed(args.output / "support_union_features.npz", **support,
        image_paths=np.asarray(observed), group_ids=np.asarray([r["group_id"] for r in job["union"]]),
        labels=np.asarray([r["labels"] for r in job["union"]], dtype=np.int64),
        class_names=np.asarray(job["classes"]))
    lookup = {row["image"]: i for i, row in enumerate(job["union"])}
    queries = {}
    for seed in SEEDS:
        rows = job["supports"][seed]
        indices = np.asarray([lookup[r["image"]] for r in rows])
        fit = fit_query_directions(support["class_local"][indices], [r["labels"] for r in rows],
                                   original, etas=(0., .3))
        queries[seed] = torch.as_tensor(fit["queries"][1], device=matcher.device)
        np.savez_compressed(args.output / f"seed_{seed}/query_fit.npz", **fit,
                            support_union_indices=indices)
        write_json(args.output / f"seed_{seed}/support_manifest.json", dict(
            support_seed=seed, support_rows=rows, eta=.3, test_labels_used=False))
    progress("support_fitted", support_union_count=len(observed))

    predictions = {s: {b: [] for b in BRANCHES} for s in SEEDS}
    maps = {s: {b: [] for b in BRANCHES} for s in SEEDS}
    test_features = {"class_local": [], "logits": []}
    checks = {"first_batch": {}, "all_batch_native_max_abs_error": {}}
    includes_cls = bool(evaluator.model.radzero_head_enable_cross_attend_cls_token)
    offset, grid, observed = 0, None, []

    def compare(name, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        error = float((actual - expected).abs().max().cpu())
        checks["all_batch_native_max_abs_error"][name] = max(
            checks["all_batch_native_max_abs_error"].get(name, 0.), error)

    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test"]:
            stop = offset + len(batch["image_paths"])
            if batch["image_paths"] != [r["image"] for r in job["test"][offset:stop]]:
                raise ValueError("Test loader reordered images")
            if tuple(batch["images"].shape[-2:]) != (518, 518):
                raise ValueError("Native test image input is not 518x518")
            vision = matcher.encode_vision(batch["images"])
            native_original = matcher.match_prepared(vision, original_tensor, return_maps=True, return_features=True)
            for seed in SEEDS:
                values = evaluate_interventions(evaluator.model, vision, original_tensor, queries[seed])
                if offset == 0:
                    checks["first_batch"][str(seed)] = audit_native_equivalence(
                        evaluator.model, vision, original_tensor, queries[seed])
                native_query = matcher.match_prepared(vision, queries[seed], return_maps=True, return_features=True)
                for branch, native in (("original", native_original), ("query", native_query)):
                    for key in ("logits", "grounding_map", "grounding_per_head", "class_local"):
                        compare(f"seed_{seed}.{branch}.{key}", values[branch][key], native[key])
                for a, b in (("original", "readout_only"), ("aggregation_only", "query")):
                    torch.testing.assert_close(values[a]["grounding_map"], values[b]["grounding_map"], atol=0, rtol=0)
                for branch in BRANCHES:
                    predictions[seed][branch].append(torch.sigmoid(values[branch]["logits"]).cpu().numpy())
                    value = values[branch]["grounding_map"].cpu().numpy()
                    grid = int(round((value.shape[-1] - int(includes_cls)) ** .5))
                    if grid != 37 or grid * grid != value.shape[-1] - int(includes_cls):
                        raise ValueError("Native spatial map is not a 37x37 patch grid")
                    maps[seed][branch].append(value)
            for key in test_features:
                test_features[key].append(native_original[key].cpu().numpy())
            observed.extend(batch["image_paths"])
            offset = stop
            if stop == len(job["test"]) or stop <= args.batch_size or (stop // args.batch_size) % 25 == 0:
                progress("matching_interventions", done=stop, total=len(job["test"]))
    if observed != [r["image"] for r in job["test"]]:
        raise ValueError("Incomplete test image coverage")
    meta = dict(image_paths=np.asarray(observed), class_names=np.asarray(job["classes"]),
                group_ids=np.asarray([r["group_id"] for r in job["test"]]))
    np.savez_compressed(args.output / "test_features.npz", **meta,
                       **{key: np.concatenate(parts) for key, parts in test_features.items()})
    artifacts = ["support_union_features.npz", "test_features.npz", "support_union.json",
                 "source_sha256.json", "input_geometry.json", "eval_config.json"]
    seed_artifacts = {}
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        p = {b: np.concatenate(predictions[seed][b]) for b in BRANCHES}
        m = {b: np.concatenate(maps[seed][b]) for b in BRANCHES}
        if any(value.shape != (len(observed), 1) or not np.isfinite(value).all() for value in p.values()):
            raise ValueError("Malformed single-class test predictions")
        np.savez_compressed(folder / "predictions.npz", **meta, **p)
        np.savez_compressed(folder / "native_maps.npz", **meta, includes_cls=np.asarray(includes_cls),
                            grid_size=np.asarray(grid), **m)
        names = ["predictions.npz", "native_maps.npz", "query_fit.npz", "support_manifest.json"]
        seed_artifacts[str(seed)] = {name: sha(folder / name) for name in names}
        artifacts.extend(f"seed_{seed}/{name}" for name in names)
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Backbone parameters are not frozen")
    write_json(args.output / "native_parity.json", checks)
    artifacts.append("native_parity.json")
    for path, digest in sources.items():
        checked(path, digest)
    status = "smoke_complete" if args.smoke else "complete"
    result = dict(status=status, model="AlphaRAD", dataset="siim_new_holdout", smoke=args.smoke,
        support_seeds=list(SEEDS), branches=list(BRANCHES), test_count=len(observed),
        full_test_count=len(job["test_all"]), support_union_count=len(job["union"]),
        eta=.3, views=1, protocol_path=str(args.protocol.resolve()), protocol_sha256=sha(args.protocol),
        source_sha256=sources, artifact_sha256={name: sha(args.output / name) for name in artifacts},
        seed_artifacts=seed_artifacts, checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored,
        test_labels_read=False, test_masks_read=False, test_rle_read=False, frozen_model=True,
        fresh_encode_all_test_images=True, positive_prompts=prompts, native_test_transform=native_transform,
        spatial_geometry="512x512 source PNG -> native 518x518 resize; square center crop preserves coverage; 37x37 patch map",
        grounding_definition="Mean per-head scaled cosine of attention query and frozen image keys; not classification attribution",
        spatial_equalities="Original=readout_only; aggregation_only=query, by construction",
        cohort_role="SIIM DICOM header groups absent from audited historical usage manifests; longitudinal patient uniqueness is unverified; not a new hospital cohort",
        gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"), physical_gpu=args.gpu,
        elapsed_seconds=time.time()-started)
    write_json(args.output / "result.json", result)
    progress(status, done=len(observed), total=len(job["test"]))


def run_owned(args):
    # Ownership is acquired before entering the failure handler. An existing
    # output raises without writing failure.json into someone else's results.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output / "failure.json", dict(error=str(exc), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.gpu not in (4, 5, 6, 7) or args.smoke < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("Use GPU 4--7, positive batch size, and nonnegative smoke/workers")
    args.output, args.protocol = args.output.resolve(), args.protocol.resolve()
    run_owned(args)


if __name__ == "__main__":
    main()
