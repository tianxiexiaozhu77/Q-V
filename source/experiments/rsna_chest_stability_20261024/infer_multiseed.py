#!/usr/bin/env python3
"""Fixed-cohort, five-support-seed inference with frozen BiomedCLIP / AlphaRAD.

Only selected support images are encoded. Every test view is encoded once and
matched against the five support-derived queries. This worker never chooses a
seed, query strength, view, or threshold using evaluation outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

MEDICAL = Path(__file__).resolve().parents[2]
ALPHA = MEDICAL / "AlphaRAD"
BASELINES = MEDICAL / "baselines"
PHASE6 = MEDICAL / "experiments/query_search_20261021_152010/code_phase6"
OLD_PROTOCOL = MEDICAL / "experiments/fewshot_native_query/protocol_v1"
SEEDS = (42, 43, 44, 45, 46)
BRANCHES = ("original", "query_refined", "tta_center_original", "tta_center_refined")
CHECKPOINT = ALPHA / "local_assets/model/xraydinov2_mpnet/eccv_mpnet_518.tar"
CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    import numpy as np
    def default(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=default) + "\n")
    temporary.replace(path)


def canonical(path):
    return str(Path(path).resolve())


def _read_json_spec(spec, root, sources):
    if isinstance(spec, str):
        spec = {"path": spec}
    path = Path(spec.get("path", spec.get("manifest", "")))
    if not path.is_absolute():
        path = root / path
    digest = sha(path)
    if spec.get("sha256") and spec["sha256"] != digest:
        raise ValueError("Frozen protocol source changed: " + str(path))
    sources[str(path)] = digest
    return json.loads(path.read_text())


def load_protocol(path, dataset):
    """Read a frozen dataset protocol; support/test grouping is checked again."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "catalog.json"
    sources = {str(path): sha(path)}
    config = json.loads(path.read_text())
    if "datasets" in config:
        child = config["datasets"][dataset]
        if isinstance(child, str) or "path" in child or "manifest" in child:
            config = _read_json_spec(child, path.parent, sources)
        else:
            config = child
    if config["dataset"] != dataset:
        raise ValueError("Dataset argument and frozen protocol differ")
    if float(config.get("eta", .3)) != .3 or float(config.get("crop_fraction", .8)) != .8:
        raise ValueError("This stability experiment fixes eta=0.3, center crop=0.8")
    classes = config["class_names"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Invalid ordered class names")

    def normalized(row):
        image = canonical(row.get("image") or row["image_path"])
        labels = row.get("labels")
        if not labels or len(labels) != len(classes) or any(float(v) not in (0., 1.) for v in labels):
            raise ValueError("Expected one ordered binary label vector per image")
        pid = str(row.get("patient_id") or "")
        group = str(row.get("group_id") or row.get("group") or pid)
        if not group:
            raise ValueError("Protocol must provide audited group identity; worker does not infer patient IDs")
        if dataset == "rsna_pneumonia" and not re.fullmatch(r"NIH:\d{8}", pid):
            raise ValueError("RSNA requires the mapped NIH:######## patient identifier, not the per-image DICOM UUID")
        if not Path(image).is_file():
            raise FileNotFoundError(image)
        return dict(image_path=image, labels=[int(v) for v in labels], group_id=group,
                    patient_id=pid, byte_sha256=row.get("byte_sha256"))

    # Authenticate immutable evaluation labels even when the dataset protocol
    # also includes rows with separately audited group identifiers.
    test_manifest = _read_json_spec(config["test_manifest"], path.parent, sources)
    rows = config.get("test_rows", test_manifest["rows"])
    test = [normalized(row) for row in rows]
    source_signature = [(canonical(row.get("image") or row["image_path"]), row["labels"])
                        for row in test_manifest["rows"]]
    if [(row["image_path"], row["labels"]) for row in test] != source_signature:
        raise ValueError("Evaluation identities or labels differ from frozen source manifest")
    if len({r["image_path"] for r in test}) != len(test):
        raise ValueError("Duplicate fixed test images")
    test_paths = {r["image_path"] for r in test}
    test_groups = {r["group_id"] for r in test}
    test_patients = {r["patient_id"] for r in test if r["patient_id"]}
    supports, union = {}, {}
    for seed in SEEDS:
        seed_entry = config["seeds"][str(seed)]
        raw = seed_entry.get("support_rows")
        if raw is None:
            manifest = _read_json_spec(seed_entry["support_manifest"], path.parent, sources)
            raw = manifest.get("support_rows", manifest.get("rows"))
        selected = [normalized(row) for row in raw]
        expected_count = 10 if dataset == "rsna_pneumonia" else 50
        if len(selected) != expected_count or len({r["image_path"] for r in selected}) != len(selected):
            raise ValueError("Unexpected support count or duplicate supports")
        if len({r["group_id"] for r in selected}) != len(selected):
            raise ValueError("Each support image must represent a distinct audited patient/image group")
        if ({r["image_path"] for r in selected} & test_paths or
                {r["group_id"] for r in selected} & test_groups or
                {r["patient_id"] for r in selected if r["patient_id"]} & test_patients):
            raise ValueError("Support/test image, group, or known patient overlap")
        for row in selected:
            if row["byte_sha256"] and sha(row["image_path"]) != row["byte_sha256"]:
                raise ValueError("Selected support bytes changed")
            if row["image_path"] in union and union[row["image_path"]] != row:
                raise ValueError("Cross-seed support metadata conflict")
            union[row["image_path"]] = row
        supports[seed] = selected
    # Sorted union is solely an image-forward cache; every fit uses its own
    # original ordered support rows and full labels, never other seeds' labels.
    selected_union = [union[key] for key in sorted(union)]
    old_path = OLD_PROTOCOL / (dataset + ".json")
    sources[str(old_path)] = sha(old_path)
    old = json.loads(old_path.read_text())
    if classes != old["class_names"] or [r["image_path"] for r in test] != [canonical(v) for v in old["test_images"]]:
        raise ValueError("Class order or fixed evaluation cohort changed from first round")
    spatial = []
    if dataset == "chestxdet10":
        spec = config.get("spatial_manifest", old["spatial_manifest"])
        spatial = _read_json_spec(spec, path.parent, sources)["rows"]
        if len(spatial) != 409 or any(canonical(r["image"]) not in test_paths for r in spatial):
            raise ValueError("Native ChestXDet10 spatial holdout must retain its 409 images")
    return dict(config=config, path=path, sources=sources, classes=classes, test=test,
                supports=supports, union=selected_union, old=old, spatial_rows=spatial)


def progress(args, stage, started, done=0, total=0):
    record = dict(status=stage, model=args.model, dataset=args.dataset,
                  done=done, total=total, elapsed_seconds=time.time() - started,
                  seeds=list(SEEDS), smoke=args.smoke)
    write(args.output / "status.json", record)
    print(json.dumps(record), flush=True)


def historical_parity(model, protocol, rows, arrays):
    """Unchanged branches are controls; historical Q is not a target to fit."""
    import numpy as np
    reference = protocol["old"]["references"][model]
    if sha(reference["path"]) != reference["sha256"]:
        raise ValueError("Historical reference result changed")
    root = Path(reference["path"]).parent
    if model == "AlphaRAD":
        paths = json.loads((root / "image_paths.json").read_text())
        expected = {branch: np.load(root / "candidates" / branch / "probabilities.npy", allow_pickle=False)
                    for branch in ("original", "tta_center_original")}
    else:
        with np.load(root / "predictions.npz", allow_pickle=False) as old:
            paths = old["image_paths"].tolist()
            if old["class_names"].tolist() != protocol["classes"]:
                raise ValueError("Historical BiomedCLIP class order changed")
            expected = {branch: old[branch] for branch in ("original", "tta_center_original")}
    lookup = {canonical(p): i for i, p in enumerate(paths)}
    ids = [lookup[row["image_path"]] for row in rows]
    checks = {}
    for branch, values in expected.items():
        actual, old = arrays[SEEDS[0]][branch], values[ids]
        np.testing.assert_allclose(actual, old, atol=2e-5, rtol=2e-5)
        checks[branch] = dict(passed=True, max_abs_error=float(np.max(np.abs(actual - old))),
                              bitwise_equal=bool(np.array_equal(actual, old)))
    return dict(passed=True, checks=checks, reference=reference,
                images=len(rows), atol=2e-5, rtol=2e-5)


def biomed_inference(args, protocol, rows, started):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, str(BASELINES / "local_setup"))
    from query_refinement_common import fit_directions, refine_query, center_pixels
    adapter = BASELINES / "protocols/alpharad_binary/BiomedCLIP/local_setup/adapter.py"
    spec = importlib.util.spec_from_file_location("_stability_biomed_native", adapter)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    progress(args, "loading", started)
    native = module.create_adapter("cuda")
    native.model.requires_grad_(False).eval()
    classes = protocol["classes"]
    count = len(classes)
    phrases = [name.replace("_", " ") for name in classes]
    prompts = [f"There is no {name}." for name in phrases] + [f"There is {name}." for name in phrases]
    with torch.inference_mode():
        text = native.model.encode_text(native.tokenizer(prompts).to(native.device), normalize=True)
    negative, positive = text[:count], text[count:]
    native._binary_text[tuple(phrases)] = text
    def pixels(batch):
        output = []
        for row in batch:
            with Image.open(row["image_path"]) as image:
                output.append(native.preprocess(image.convert("RGB")))
        return torch.stack(output).to(native.device)
    def score(features, pos):
        similarities = features @ torch.cat((negative, pos)).T
        return torch.stack((similarities[:, :count], similarities[:, count:]), dim=-1).softmax(-1)[..., 1]
    parts = []
    with torch.inference_mode():
        for offset in range(0, len(protocol["union"]), args.batch_size):
            parts.append(native.model.encode_image(pixels(protocol["union"][offset:offset + args.batch_size]),
                                                   normalize=True).float().cpu().numpy())
    features = np.concatenate(parts)
    np.save(args.output / "support_union_features.npy", features, allow_pickle=False)
    lookup = {row["image_path"]: i for i, row in enumerate(protocol["union"])}
    queries = {}
    for seed in SEEDS:
        supports = protocol["supports"][seed]
        ids = [lookup[row["image_path"]] for row in supports]
        fit = fit_directions(features[ids], [row["labels"] for row in supports])
        queries[seed] = refine_query(positive, fit["directions"], .3, preserve_norm=True)
        np.savez_compressed(args.output / f"seed_{seed}/query_fit.npz",
            **{key: value.cpu().numpy() for key, value in fit.items()},
            original_queries=positive.cpu().numpy(), original_negative_queries=negative.cpu().numpy(),
            queries=queries[seed].cpu().numpy(), support_union_indices=np.asarray(ids))
    chunks = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    parity = {}
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            x = pixels(batch)
            identity = native.model.encode_image(x, normalize=True).float()
            center = native.model.encode_image(center_pixels(x, "center80"), normalize=True).float()
            original, centered = score(identity, positive), score(center, positive)
            if offset == 0:
                expected = native.predict([row["image_path"] for row in batch], classes)
                np.testing.assert_allclose(original.cpu().numpy(), expected, atol=2e-5, rtol=2e-5)
                parity["native_adapter"] = dict(passed=True, max_abs_error=float(np.max(np.abs(original.cpu().numpy()-expected))))
            for seed in SEEDS:
                q, qc = score(identity, queries[seed]), score(center, queries[seed])
                values = dict(original=original, query_refined=q,
                              tta_center_original=(original + centered) * .5,
                              tta_center_refined=(q + qc) * .5)
                for branch, value in values.items():
                    chunks[seed][branch].append(value.cpu().numpy())
            if offset == 0 or offset // args.batch_size % 25 == 0 or offset + len(batch) == len(rows):
                progress(args, "inference", started, offset + len(batch), len(rows))
    if any(parameter.requires_grad for parameter in native.model.parameters()):
        raise ValueError("BiomedCLIP must remain frozen")
    for path in (adapter, BASELINES / "local_setup/query_refinement_common.py"):
        protocol["sources"][str(path)] = sha(path)
    return chunks, {}, [], parity, dict(native_metadata=native.metadata,
        classification_aggregation="mean native binary-softmax probabilities", temperature=1.,
        query_norm="preserve original norm", spatial="no verified native spatial evaluation")


def alpha_inference(args, protocol, rows, started):
    import numpy as np
    import torch
    sys.path[:0] = [str(ALPHA), str(ALPHA / "local_setup"), str(PHASE6)]
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator, fit_query_directions
    from eval_query_refinement import original_helper_check
    from view_geometry import apply_view, aggregate_outputs
    seed_everything(1234)
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Published AlphaRAD checkpoint changed")
    for name, entries in (("support_union", protocol["union"]), ("test", rows)):
        with (args.output / f"{name}_inputs.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image"])
            writer.writerows([[row["image_path"]] for row in entries])
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   pin_memory=True, shuffle=False, drop_last=False)
    if args.workers:
        options["prefetch_factor"] = 2
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(CHECKPOINT)]},
        "data_test": {name: {"name": name, "data_type": "image_classification",
            "data_path": str(args.output / f"{name}_inputs.csv"), "n_class": len(protocol["classes"]),
            "normalize": None} for name in ("support_union", "test")}, "dataloader": {"test": options}}
    write(args.output / "eval_config.json", config)
    progress(args, "loading", started)
    evaluator = Evaluator(config, [str(CHECKPOINT)], "test")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    original, prompts = original_queries_from_evaluator(evaluator, protocol["classes"])
    matcher = QueryRefinementMatcher(evaluator.model, original[None], (0.,),
                                    use_all_image_tokens=evaluator.use_all_image_tokens)
    observed, parts = [], []
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["support_union"]:
            vision = matcher.encode_vision(batch["images"])
            parts.append(matcher.match_prepared(vision, original, return_features=True)["class_local"].cpu().numpy())
            observed.extend(batch["image_paths"])
    if [canonical(p) for p in observed] != [r["image_path"] for r in protocol["union"]]:
        raise ValueError("Support union image order changed")
    features = np.concatenate(parts)
    np.save(args.output / "support_union_features.npy", features, allow_pickle=False)
    lookup = {row["image_path"]: i for i, row in enumerate(protocol["union"])}
    queries = {}
    for seed in SEEDS:
        supports = protocol["supports"][seed]
        ids = [lookup[row["image_path"]] for row in supports]
        fit = fit_query_directions(features[ids], [row["labels"] for row in supports], original, etas=(0., .3))
        queries[seed] = fit["queries"][1]
        np.savez_compressed(args.output / f"seed_{seed}/query_fit.npz", **fit,
                            support_union_indices=np.asarray(ids))
    chunks = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    map_chunks = {seed: {key: [] for key in BRANCHES} for seed in SEEDS}
    spatial_by_image = {canonical(row["image"]): row for row in protocol["spatial_rows"]}
    spatial_rows = []
    parity, offset = {}, 0
    with torch.inference_mode():
        for batch in evaluator.data_loader_dict["test"]:
            stop = offset + len(batch["images"])
            if [canonical(p) for p in batch["image_paths"]] != [r["image_path"] for r in rows[offset:stop]]:
                raise ValueError("Test image order changed")
            selected = [j for j, path in enumerate(batch["image_paths"]) if canonical(path) in spatial_by_image]
            spatial_rows.extend(spatial_by_image[canonical(batch["image_paths"][j])] for j in selected)
            maps_enabled = bool(selected)
            identity = matcher.encode_vision(batch["images"])
            # Preserve first-round tensor preprocessing: center crop on CPU,
            # then the same FP32 encoder; no new augmentation implementation.
            center = matcher.encode_vision(apply_view(batch["images"], "center80"))
            base_outputs = {"identity": matcher.match_prepared(identity, original, return_maps=maps_enabled),
                            "center80": matcher.match_prepared(center, original, return_maps=maps_enabled)}
            base_view = aggregate_outputs(base_outputs, ["identity", "center80"])
            if offset == 0:
                parity["native_author_helpers"] = original_helper_check(evaluator, matcher, batch, protocol["classes"])
            for seed in SEEDS:
                outputs = {"identity": matcher.match_prepared(identity, queries[seed], return_maps=maps_enabled),
                           "center80": matcher.match_prepared(center, queries[seed], return_maps=maps_enabled)}
                values = dict(original=base_outputs["identity"], query_refined=outputs["identity"],
                              tta_center_original=base_view,
                              tta_center_refined=aggregate_outputs(outputs, ["identity", "center80"]))
                for branch, value in values.items():
                    chunks[seed][branch].append(torch.sigmoid(value["logits"]).cpu().numpy())
                    if maps_enabled:
                        map_chunks[seed][branch].append(value["grounding_map"][selected].cpu().numpy())
            if offset == 0 or offset // args.batch_size % 25 == 0 or stop == len(rows):
                progress(args, "inference", started, stop, len(rows))
            offset = stop
    if any(parameter.requires_grad for parameter in evaluator.model.parameters()):
        raise ValueError("AlphaRAD must remain frozen")
    for path in (ALPHA / "local_setup/query_refinement.py", ALPHA / "local_setup/eval_query_refinement.py",
                 PHASE6 / "view_geometry.py", ALPHA / "cxrclip/evaluator_utils.py"):
        protocol["sources"][str(path)] = sha(path)
    return chunks, map_chunks, spatial_rows, parity, dict(checkpoint_sha256=CHECKPOINT_SHA,
        ignored_checkpoint_keys=ignored, prompts=prompts, classification_aggregation="mean native logits then sigmoid",
        support_feature="native original-query class-local FLaS context", query_norm="unit at eta=0.3",
        spatial="native scaled FLaS map; inverse warp center; mean over valid coverage")


def spatial_hits(maps, rows, classes):
    """Per-image/class sufficient statistics matching the author's PG code.

No threshold is fitted. Peak interpolation and coordinate rounding are copied
from cxrclip.evaluator_utils.pointing_game and checked against that function.
"""
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    maps = torch.as_tensor(maps, dtype=torch.float32)
    if maps.shape != (len(rows), len(classes), 1370):
        raise ValueError("Expected native CLS + 37x37 patch map")
    patches = maps[..., 1:].reshape(len(rows), len(classes), 37, 37)
    hits = np.zeros((len(rows), len(classes)), dtype=np.uint8)
    cases = np.zeros_like(hits)
    for i, row in enumerate(rows):
        image = cv2.imread(row["image"])
        if image is None:
            raise FileNotFoundError(row["image"])
        height, width = image.shape[:2]
        for j, cls in enumerate(classes):
            boxes = [box for box, name in zip(row["boxes"], row["class_names_present"]) if name == cls]
            if not boxes:
                continue
            heat = patches[i, j][None, None]
            if height == width:
                heat = F.interpolate(heat, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
                index = int(torch.argmax(heat).item())
                y_peak, x_peak = index // width, index % width
            else:
                scale = 518 / float(min(height, width))
                resized_h, resized_w = int(round(height * scale)), int(round(width * scale))
                top, left = max((resized_h - 518) // 2, 0), max((resized_w - 518) // 2, 0)
                heat = F.interpolate(heat, size=(518, 518), mode="bilinear", align_corners=False)[0, 0]
                index = int(torch.argmax(heat).item())
                y_peak = int(round((index // 518 + top) / scale))
                x_peak = int(round((index % 518 + left) / scale))
            cases[i, j] = 1
            hits[i, j] = int(any(box[0] <= x_peak <= box[2] and box[1] <= y_peak <= box[3] for box in boxes))
    totals = cases.sum(0)
    per_class = {cls: float(hits[:, j].sum() / totals[j]) for j, cls in enumerate(classes) if totals[j]}
    return hits, cases, dict(pointing_game=float(np.mean(list(per_class.values()))),
                             per_class=per_class, cases=totals.tolist())


def save_spatial(args, protocol, maps, spatial_rows, rows):
    import numpy as np
    from cxrclip.evaluator_utils import gather_pointinggame_statistics, pointing_game_score
    row_lookup = {row["image_path"]: row for row in rows}
    identities = dict(image_paths=np.asarray([canonical(row["image"]) for row in spatial_rows]),
                      group_ids=np.asarray([row_lookup[canonical(row["image"])]["group_id"] for row in spatial_rows]),
                      class_names=np.asarray(protocol["classes"]))
    outputs = {}
    for seed in SEEDS:
        joined = {branch: np.concatenate(parts) for branch, parts in maps[seed].items()}
        values, metrics = {}, {}
        for branch, array in joined.items():
            hits, cases, metrics[branch] = spatial_hits(array, spatial_rows, protocol["classes"])
            values[branch + "_hits"], values[branch + "_cases"] = hits, cases
            # Validate both unchanged and changed query output against author
            # routines on the first seed; other seeds use identical statistic.
            if seed == SEEDS[0]:
                stats = gather_pointinggame_statistics(protocol["classes"], array,
                    [r["boxes"] for r in spatial_rows], [r["class_names_present"] for r in spatial_rows],
                    [r["image"] for r in spatial_rows], grounding_type="pointing_game")
                expected = pointing_game_score(stats, _print=False)
                np.testing.assert_allclose(metrics[branch]["pointing_game"], expected["Pointing_Game(Avg)"], atol=1e-12, rtol=0)
                for cls, value in metrics[branch]["per_class"].items():
                    np.testing.assert_allclose(value, expected[cls]["pointing_game"], atol=1e-12, rtol=0)
                metrics[branch]["author_helper_parity"] = True
        if not args.smoke:
            reference = protocol["old"]["spatial_manifest"]["reference"]
            if sha(reference["result"]) != reference["sha256"]:
                raise ValueError("Historical native PG reference changed")
            old = json.loads(Path(reference["result"]).read_text())
            for branch in ("original", "tta_center_original"):
                np.testing.assert_allclose(metrics[branch]["pointing_game"],
                    old["candidates"][branch]["metrics"]["pointing_game"], atol=1e-12, rtol=0)
        folder = args.output / f"seed_{seed}"
        np.savez_compressed(folder / "native_maps.npz", **identities, **joined)
        np.savez_compressed(folder / "spatial.npz", **identities, **values)
        write(folder / "spatial_result.json", dict(rows=len(spatial_rows), branches=metrics,
            metric="macro class pointing-game hit rate", group_unit="image/exact-duplicate cluster; patient identities unavailable"))
        outputs[str(seed)] = metrics
    return outputs


def run(args):
    import numpy as np
    import torch
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(args.protocol, args.dataset)
    protocol["sources"][str(Path(__file__).resolve())] = sha(__file__)
    rows = protocol["test"][:args.smoke] if args.smoke else protocol["test"]
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        folder.mkdir()
        write(folder / "support_manifest.json", dict(seed=seed, support_rows=protocol["supports"][seed],
            eta=.3, test_labels_used_for_fit=False))
    write(args.output / "support_union.json", protocol["union"])
    write(args.output / "input_audit.json", dict(protocol=str(protocol["path"]),
        protocol_sha256=sha(protocol["path"]), model=args.model, dataset=args.dataset,
        support_union_count=len(protocol["union"]), support_count_by_seed={str(s): len(protocol["supports"][s]) for s in SEEDS},
        test_count=len(protocol["test"]), inference_count=len(rows), classes=protocol["classes"],
        groups=len({r["group_id"] for r in rows}), support_test_group_overlap=0,
        all_seeds_same_fixed_test=True))
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Assigned CUDA device unavailable")
    function = alpha_inference if args.model == "AlphaRAD" else biomed_inference
    chunks, maps, spatial_rows, parity, metadata = function(args, protocol, rows, started)
    arrays = {seed: {key: np.concatenate(parts) for key, parts in branch.items()} for seed, branch in chunks.items()}
    labels = np.asarray([row["labels"] for row in rows], dtype=np.uint8)
    for seed, branches in arrays.items():
        for key, values in branches.items():
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError("Malformed prediction array")
        for key in ("original", "tta_center_original"):
            np.testing.assert_array_equal(branches[key], arrays[SEEDS[0]][key])
    parity["historical_unchanged_arms"] = historical_parity(args.model, protocol, rows, arrays)
    parity["unchanged_arms_bitwise_equal_across_seeds"] = True
    write(args.output / "native_parity.json", parity)
    sys.path.insert(0, str(BASELINES / "local_setup"))
    from evaluate_baseline_buffered import classification_metrics
    spatial = save_spatial(args, protocol, maps, spatial_rows, rows) if spatial_rows else {}
    results = {}
    for seed in SEEDS:
        folder = args.output / f"seed_{seed}"
        identity = dict(labels=labels, image_paths=np.asarray([r["image_path"] for r in rows]),
                        group_ids=np.asarray([r["group_id"] for r in rows]),
                        class_names=np.asarray(protocol["classes"]), seed=np.asarray(seed))
        if args.dataset == "rsna_pneumonia":
            identity["patient_ids"] = np.asarray([r["patient_id"] for r in rows])
        np.savez_compressed(folder / "predictions.npz", **identity, **arrays[seed])
        result = dict(status="smoke_complete" if args.smoke else "complete", model=args.model,
            dataset=args.dataset, seed=seed, rows=len(rows), groups=len(set(r["group_id"] for r in rows)),
            branches={key: classification_metrics(labels, value, protocol["classes"]) for key, value in arrays[seed].items()},
            spatial=spatial.get(str(seed)), eta=.3, views=["identity", "center80"], metadata=metadata,
            same_fixed_test_all_seeds=True, no_test_selection=True,
            prediction_sha256=sha(folder / "predictions.npz"))
        write(folder / "result.json", result)
        results[str(seed)] = result
    changed = [path for path, digest in protocol["sources"].items() if sha(path) != digest]
    if changed:
        raise ValueError("Protocol/source changed during inference: " + repr(changed))
    write(args.output / "source_sha256.json", protocol["sources"])
    status = "smoke_complete" if args.smoke else "complete"
    write(args.output / "result.json", dict(status=status, model=args.model, dataset=args.dataset,
        seeds=list(SEEDS), rows=len(rows), seed_results=results, native_parity=parity, metadata=metadata,
        no_gradient_training=True, batch_size=args.batch_size, elapsed_seconds=time.time()-started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_version=torch.__version__))
    progress(args, status, started, len(rows), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("BiomedCLIP", "AlphaRAD"))
    parser.add_argument("--dataset", required=True, choices=("rsna_pneumonia", "chestxdet10"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    args.batch_size = args.batch_size or (16 if args.model == "BiomedCLIP" else 8)
    if args.batch_size < 1 or args.workers < 0 or args.smoke < 0:
        parser.error("Invalid batch, workers, or smoke")
    args.output = args.output.resolve()
    if args.output.exists():
        # Do not overwrite even failure metadata in a historical output folder.
        raise FileExistsError("Use a new output directory: " + str(args.output))
    try:
        run(args)
    except BaseException as exc:
        if args.output.is_dir():
            write(args.output / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
