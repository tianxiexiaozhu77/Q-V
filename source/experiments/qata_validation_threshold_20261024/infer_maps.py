#!/usr/bin/env python3
"""Frozen AlphaRAD QaTa validation maps and authenticated phase6 test cache.

No segmentation masks are decoded and no threshold/metric is computed here.
The old original/refined queries, preprocessing, and center-view geometry remain
unchanged. Only the new validation images need full inference; eight test images
are forwarded without annotations to audit all four historical cached branches.
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

MEDICAL = Path(__file__).resolve().parents[2]
ALPHA = MEDICAL / "AlphaRAD"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
OLD_RUN = MEDICAL / "experiments/query_search_20261021_152010/runs/phase6/alpharad_qatacovid_seg"
FORMAL = ALPHA / "runs/query_refinement_20261021_1441/qatacovid_seg"
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
HISTORICAL = dict(zip(BRANCHES, ("original", "uniform_eta03", "tta_center_original", "tta_center_refined")))
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def checked(path, expected, sources):
    path = Path(path).resolve()
    actual = sha(path)
    if expected and actual != expected:
        raise ValueError("Frozen source changed: " + str(path))
    sources[str(path)] = actual
    return path


def load_protocol(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "protocol.json"
    sources = {str(path): sha(path)}
    protocol = json.loads(path.read_text())
    if "datasets" in protocol:
        child = protocol["datasets"]["qatacovid_seg"]
        spec = child if isinstance(child, dict) else {"path": child}
        child = Path(spec["path"])
        if not child.is_absolute():
            child = path.parent / child
        path = checked(child, spec.get("sha256"), sources)
        protocol = json.loads(path.read_text())
    if (protocol["dataset"] != "qatacovid_seg" or protocol["class_names"] != ["covid"]
            or protocol.get("eta") != .3 or protocol.get("crop_fraction") != .8):
        raise ValueError("Expected QaTa, covid query, eta=.3 and center80 fixed protocol")
    rows = {}
    for split in ("validation", "test", "support"):
        spec = protocol[split]
        rows_path = Path(spec.get("rows_path", protocol.get(split + "_rows_path", "")))
        if not rows_path.is_absolute():
            rows_path = path.parent / rows_path
        checked(rows_path, spec.get("rows_sha256", protocol.get(split + "_rows_sha256")), sources)
        if split == "test":
            identity_path = Path(protocol["test_identities_path"])
            if not identity_path.is_absolute():
                identity_path = path.parent / identity_path
            checked(identity_path, protocol["test_identities_sha256"], sources)
            values = json.loads(identity_path.read_text())
            if any(set(row) & {"class", "label", "labels", "mask", "mask_path", "EncodedPixels"}
                   for row in values):
                raise ValueError("The test identity-only manifest must contain no annotations")
        else:
            values = json.loads(rows_path.read_text())
        if isinstance(values, dict):
            values = values["rows"]
        rows[split] = values
        if len(values) != spec["rows_count"]:
            raise ValueError(split + " row count differs from protocol")
        checked(spec["csv_path"], spec["csv_sha256"], sources)
        images = [str(Path(v.get("image", v.get("image_path"))).resolve()) for v in values]
        if len(images) != len(set(images)):
            raise ValueError("Repeated image in " + split)
        for row, image in zip(values, images):
            row["image_path"] = image
            if not Path(image).is_file():
                raise FileNotFoundError(image)
            if split != "support" and (not row.get("patient_id") or not row.get("group_id")):
                raise ValueError("Known patient/group identity required for validation and test")
            image_digest = row.get("image_sha256", row.get("byte_sha256"))
            if image_digest and sha(image) != image_digest:
                raise ValueError("Protocol image content changed: " + image)
    if len(rows["test"]) != 2113 or len(rows["support"]) != 10:
        raise ValueError("Historical fixed test/support counts changed")
    for first, second in (("validation", "test"), ("support", "test"), ("support", "validation")):
        images_a = {v["image_path"] for v in rows[first]}
        images_b = {v["image_path"] for v in rows[second]}
        patients_a = {v["patient_id"] for v in rows[first] if v.get("patient_id")}
        patients_b = {v["patient_id"] for v in rows[second] if v.get("patient_id")}
        if images_a & images_b or patients_a & patients_b:
            raise ValueError("Known patient/image overlap between " + first + " and " + second)
    return dict(path=path, config=protocol, rows=rows, sources=sources)


def load_historical(bundle):
    import numpy as np
    sources, protocol = bundle["sources"], bundle["config"]
    audit_path = checked(OLD_RUN / "data_audit.json", None, sources)
    audit = json.loads(audit_path.read_text())
    for path, digest in audit["source_sha256"].items():
        checked(path, digest, sources)
    for path, digest in audit["original_sources_verified"].items():
        checked(ALPHA / path, digest, sources)
    checked(CHECKPOINT, CHECKPOINT_SHA, sources)
    entry = audit["entry"]
    if (entry["actual_count"] != 2113 or entry["support_count"] != 10 or
            entry["class_names"] != ["covid"] or audit["support_labels"] != [[0]] * 5 + [[1]] * 5):
        raise ValueError("Historical support/class identity changed")
    for split, field in (("support", "support_csv_sha256"), ("test", "csv_sha256")):
        if protocol[split]["csv_sha256"] != entry[field]:
            raise ValueError("Protocol must preserve historical " + split + " CSV bytes")
    paths_path = checked(OLD_RUN / "image_paths.json", None, sources)
    paths = json.loads(paths_path.read_text())
    if paths != [v["image_path"] for v in bundle["rows"]["test"]]:
        raise ValueError("Fixed test image identities/order changed")
    # Historical visual cache authenticates ordered source image stat identity.
    cache_info = json.loads(checked(OLD_RUN / "vision_cache.json", None, sources).read_text())
    cache_path = checked(Path(cache_info["path"]) / "manifest.json", cache_info["manifest_sha256"], sources)
    cache = json.loads(cache_path.read_text())
    if cache["status"] != "complete" or cache["image_ids"] != paths or cache["cached_rows"] != 2113:
        raise ValueError("Historical visual cache is not a complete ordered reference")
    for image, fingerprint in zip(paths, cache["identity"]["input_file_fingerprints"]):
        stat = Path(image).stat()
        actual = dict(path=str(Path(image).resolve()), bytes=stat.st_size,
                      mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
        if actual != fingerprint:
            raise ValueError("Historical test image stat identity changed: " + image)
    adapter_meta = json.loads(checked(FORMAL / "query_adapter.json", None, sources).read_text())
    adapter_path = checked(FORMAL / "query_adapter.npz", adapter_meta["query_file_sha256"], sources)
    if (adapter_meta["eta"] != .3 or adapter_meta["checkpoint_sha256"] != CHECKPOINT_SHA
            or adapter_meta["support_csv_sha256"] != entry["support_csv_sha256"]):
        raise ValueError("Historical query identity changed")
    with np.load(adapter_path, allow_pickle=False) as adapter:
        original = adapter["original_queries"].copy()
        queries = adapter["queries"].copy()
    query_path = checked(OLD_RUN / "candidate_queries.npz", None, sources)
    with np.load(query_path, allow_pickle=False) as candidates:
        names, values = candidates["names"].tolist(), candidates["queries"]
        np.testing.assert_array_equal(values[names.index("original")], queries[0])
        np.testing.assert_array_equal(values[names.index("uniform_eta03")], queries[1])
    references, maps = {}, {}
    for branch, old_name in HISTORICAL.items():
        expected_plan = dict(views=["identity"] if branch in BRANCHES[:2] else ["identity", "center80"],
                             query="original" if branch in ("original", "tta_center_original") else "uniform_eta03")
        if audit["view_plans"][old_name] != expected_plan:
            raise ValueError("Historical view geometry/query assignment changed")
        path = checked(OLD_RUN / "candidates" / old_name / "grounding_map.npy", None, sources)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (2113, 1, 1370) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid historical raw map: " + str(path))
        references[branch] = dict(path=str(path), sha256=sha(path), shape=list(values.shape), dtype=str(values.dtype))
        maps[branch] = values
        actual_query = np.load(checked(path.parent / "query.npy", None, sources), allow_pickle=False)
        np.testing.assert_array_equal(actual_query, queries[0 if expected_plan["query"] == "original" else 1])
    return dict(original=original, queries=queries, references=references, maps=maps,
                adapter_meta=adapter_meta, audit=audit)


def progress(args, stage, started, done=0, total=0):
    record = dict(status=stage, model="AlphaRAD", dataset="qatacovid_seg",
                  done=done, total=total, smoke=args.smoke, elapsed_seconds=time.time() - started)
    write(args.output / "status.json", record)
    print(json.dumps(record), flush=True)


def run(args):
    import numpy as np
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator, fit_query_directions
    from eval_alpharad_candidates import controls_helper_check
    from view_geometry import apply_view, aggregate_outputs, VIEW_SPECS

    started = time.time()
    progress(args, "authenticating_protocol_and_cache", started)
    bundle = load_protocol(args.protocol)
    historical = load_historical(bundle)
    source_snapshot = dict(bundle["sources"])
    for source in (Path(__file__), PHASE6 / "view_geometry.py"):
        checked(source, None, source_snapshot)
    if historical["audit"]["view_geometry"]["center80"] != VIEW_SPECS["center80"]:
        raise ValueError("Center crop geometry differs from phase6")
    features = np.load(FORMAL / "support_class_local.npy", allow_pickle=False)
    fitted = fit_query_directions(features, historical["audit"]["support_labels"], historical["original"], etas=(0., .3))
    np.testing.assert_array_equal(fitted["queries"], historical["queries"])
    validation = bundle["rows"]["validation"]
    if args.smoke:
        validation = validation[:args.smoke]
    # The audit CSV has image paths only: the model loader never sees test GT.
    test_audit = bundle["rows"]["test"][:args.batch_size]
    for split, rows in (("validation", validation), ("test_cache_audit", test_audit)):
        with (args.output / (split + "_images.csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image"])
            writer.writerows([[row["image_path"]] for row in rows])
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = dict(base=dict(seed=1234), test=dict(checkpoint=[str(CHECKPOINT)]),
        data_test={split: dict(name=split, data_type="image_classification",
                              data_path=str(args.output / (split + "_images.csv")), n_class=1, normalize=None)
                   for split in ("validation", "test_cache_audit")}, dataloader=dict(test=options))
    write(args.output / "eval_config.json", config)
    seed_everything(1234)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("A controller-assigned CUDA GPU is required")
    progress(args, "loading", started)
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, ["covid"])
    np.testing.assert_array_equal(original, historical["original"])
    matcher = QueryRefinementMatcher(evaluator.model, historical["queries"], (0., .3),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)

    def infer_batch(batch):
        # Match historical phase6 precisely: transfer pixels before crop/resize.
        image = evaluator.encode_image(batch["images"])
        identity = matcher.prepare_vision_results(image)
        pixels = batch["images"].to(matcher.device, dtype=torch.float32, non_blocking=True)
        center = matcher.encode_vision(apply_view(pixels, "center80"))
        native, refined = {}, {}
        for name, vision in (("identity", identity), ("center80", center)):
            native[name] = matcher.match_prepared(vision, matcher.queries[0], return_maps=True)
            refined[name] = matcher.match_prepared(vision, matcher.queries[1], return_maps=True)
        outputs = dict(original=native["identity"], query_refined=refined["identity"],
                       tta_center_original=aggregate_outputs(native, ["identity", "center80"]),
                       tta_center_refined=aggregate_outputs(refined, ["identity", "center80"]))
        return outputs, image

    parity = {}
    arrays = {branch: np.empty((len(validation), 1, 1370), dtype=np.float32) for branch in BRANCHES}
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test_cache_audit"]:
            if batch["image_paths"] != [row["image_path"] for row in test_audit]:
                raise ValueError("Test cache audit order changed")
            outputs, image = infer_batch(batch)
            # The original author helper consumes names only; no masks/labels.
            helper_batch = dict(batch, label_names=[["covid"]] * len(test_audit))
            parity["original_author_helpers"] = controls_helper_check(evaluator, image, helper_batch,
                                                                       ["covid"], outputs["original"])
            for branch in BRANCHES:
                actual = outputs[branch]["grounding_map"].cpu().numpy()
                expected = historical["maps"][branch][:len(actual)]
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
                parity[branch] = dict(passed=True, images=len(actual),
                    max_abs_error=float(np.max(np.abs(actual - expected))),
                    bitwise_equal=bool(np.array_equal(actual, expected)))
        write(args.output / "parity_checks.json", parity)
        offset = 0
        for batch in evaluator.data_loader_dict["validation"]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != [row["image_path"] for row in validation[offset:stop]]:
                raise ValueError("Validation image order changed")
            outputs, _ = infer_batch(batch)
            for branch in BRANCHES:
                values = outputs[branch]["grounding_map"].cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite validation map")
                arrays[branch][offset:stop] = values
            offset = stop
            if offset == len(batch["images"]) or offset % (25 * args.batch_size) == 0 or offset == len(validation):
                progress(args, "validation_inference", started, offset, len(validation))
    if any(p.requires_grad for p in evaluator.model.parameters()):
        raise ValueError("Model parameters must remain frozen")
    for path, digest in source_snapshot.items():
        if sha(path) != digest:
            raise ValueError("Source modified during inference: " + path)
    if offset != len(validation):
        raise ValueError("Incomplete validation inference")
    np.savez_compressed(args.output / "native_maps.npz", **arrays,
                        image_paths=np.asarray([r["image_path"] for r in validation]),
                        patient_ids=np.asarray([r["patient_id"] for r in validation]),
                        group_ids=np.asarray([r["group_id"] for r in validation]),
                        class_names=np.asarray(["covid"]))
    test = bundle["rows"]["test"]
    cache_manifest = dict(status="complete", model="AlphaRAD", dataset="qatacovid_seg", split="test",
        format="native_grounding_map_refs_v1", branches=historical["references"],
        image_paths=[r["image_path"] for r in test], patient_ids=[r["patient_id"] for r in test],
        group_ids=[r["group_id"] for r in test], class_names=["covid"],
        checkpoint_sha256=CHECKPOINT_SHA, historical_run=str(OLD_RUN),
        protocol_path=str(bundle["path"]), protocol_sha256=sha(bundle["path"]),
        validation_source_inference=str(args.output), live_map_parity=parity,
        note="Raw cached maps only; test labels must not select thresholds.")
    write(args.output / "test_cache_manifest.json", cache_manifest)
    write(args.output / "result.json", dict(status="smoke_complete" if args.smoke else "complete",
        model="AlphaRAD", dataset="qatacovid_seg", validation_rows=len(validation), test_rows=len(test),
        branches=list(BRANCHES), eta=.3, crop_fraction=.8, support_count=10,
        query_source=str(FORMAL / "query_adapter.npz"), frozen_queries_bitwise_refit_check=True,
        checkpoint_sha256=CHECKPOINT_SHA, ignored_checkpoint_keys=ignored, prompts=prompts,
        network_weight_updates=False, masks_decoded=False, threshold_selection_performed=False,
        test_forward_scope="First batch only, image-only cache parity; remaining test maps reused",
        native_maps_sha256=sha(args.output / "native_maps.npz"),
        test_cache_manifest_sha256=sha(args.output / "test_cache_manifest.json"),
        protocol_path=str(bundle["path"]), source_sha256=source_snapshot, parity=parity,
        view_geometry={k: VIEW_SPECS[k] for k in ("identity", "center80")},
        no_patient_mapping_for_negative_support=True, elapsed_seconds=time.time() - started))
    progress(args, "smoke_complete" if args.smoke else "complete", started, len(validation), len(validation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error("Invalid inference size")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new empty output directory: " + str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except BaseException as error:
        write(args.output / "failure.json", dict(status="failed", error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
