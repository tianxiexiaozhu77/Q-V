#!/usr/bin/env python3
"""Reuse a saved AlphaRAD query adapter on prepared PNG/JPG images, without fitting.

Accept --images, a newline-separated --image-list, or a CSV --manifest with an
image column. DICOM/MHA inputs must first use the project's original converter.
The frozen author's Evaluator supplies the exact image preprocessing and FLaS.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Never allow this inference entry to fetch a model or tokenizer from the hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

from eval_pretrained import ASSETS, sha256, write_json

CHECKPOINT_SHA = "97ca9c1fb8e4c869abe9cee592d8b0440d4f1f765efa98808a15e574cb08ccb9"


def input_paths(args):
    """Resolve files without reading labels or fitting from the new inputs."""
    base = Path.cwd()
    if args.manifest:
        base = args.manifest.resolve().parent
        with args.manifest.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise ValueError("--manifest requires an image column")
            values = [row["image"] for row in reader]
        for index, value in enumerate(values):
            if value.startswith("["):
                parsed = ast.literal_eval(value)
                if not isinstance(parsed, list) or len(parsed) != 1:
                    raise ValueError("Each manifest row must identify exactly one image")
                values[index] = parsed[0]
    elif args.image_list:
        base = args.image_list.resolve().parent
        values = [line.strip() for line in args.image_list.read_text().splitlines() if line.strip()]
    else:
        values = args.images
    paths = []
    for value in values:
        path = Path(value).expanduser()
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"Use a prepared PNG/JPG image; convert DICOM/MHA first: {path}")
        paths.append(str(path))
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Input images must be nonempty and unique")
    return paths


def load_adapter(path):
    import numpy as np
    from query_refinement import VERSION, validate_etas

    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "query_adapter.npz"
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("version") != VERSION:
        raise ValueError("Unsupported query-adapter version")
    if sha256(path) != metadata["query_file_sha256"]:
        raise ValueError("Query-adapter checksum differs from its metadata")
    if metadata["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Adapter belongs to another pretrained checkpoint")
    classes, prompts = metadata["class_names"], metadata["prompts"]
    if (not isinstance(classes, list) or not classes or len(set(classes)) != len(classes)
            or any(not isinstance(c, str) or not c for c in classes)
            or not isinstance(prompts, list) or len(prompts) != len(classes)):
        raise ValueError("Invalid class names or prompt order")
    eta = float(metadata["eta"])
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("Saved adapter requires a finite positive eta")
    with np.load(path, allow_pickle=False) as archive:
        etas = validate_etas(archive["etas"])
        original = archive["original_queries"].copy()
        queries = archive["queries"].copy()
    if eta not in etas or original.shape != (len(classes), metadata["query_dimension"]):
        raise ValueError("Saved eta, query dimensions, or class count differ")
    if queries.shape != (len(etas), *original.shape):
        raise ValueError("Unexpected saved query matrix shape")
    if original.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError("Expected float32 query matrices")
    if not np.isfinite(original).all() or not np.isfinite(queries).all():
        raise ValueError("Nonfinite query features")
    np.testing.assert_array_equal(queries[etas.index(0.)], original)
    refined = queries[etas.index(eta)]
    np.testing.assert_allclose(np.linalg.norm(refined, axis=-1), 1., atol=2e-5, rtol=1e-5)
    # The fitting worker may evolve; the inference scorer must remain identical.
    scorer = Path(__file__).with_name("query_refinement.py")
    recorded = [value for key, value in metadata.get("source_sha256", {}).items()
                if Path(key).name == scorer.name]
    if len(recorded) != 1 or sha256(scorer) != recorded[0]:
        raise ValueError("Original query-refinement scorer differs from the saved adapter")
    return path, metadata_path, metadata, original, refined


def run(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from cxrclip import seed_everything
    from cxrclip.evaluator import Evaluator, _strict_load_published_checkpoint
    from cxrclip.prompt import constants
    from query_refinement import QueryRefinementMatcher, original_queries_from_evaluator

    start = time.time()
    torch.set_num_threads(4)
    seed_everything(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Retain the same cuDNN setting as the paired evaluation and original author
    # pipeline. The released ViT's patch projection uses their default backend.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No visible CUDA GPU; expose a free card or pass --device cpu")
    adapter, metadata_path, metadata, original, refined = load_adapter(args.adapter)
    if sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Pretrained checkpoint checksum mismatch")
    classes, paths = metadata["class_names"], input_paths(args)
    known = getattr(constants, str(metadata["dataset"]).upper(), None)
    if known is not None and list(known) != classes:
        raise ValueError("Saved class order differs from the author's dataset constant")
    input_csv = args.output / "inputs.csv"
    with input_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        writer.writerows([[path] for path in paths])
    loader = {"batch_size": args.batch_size, "num_workers": args.workers,
              "shuffle": False, "drop_last": False, "pin_memory": args.device == "cuda"}
    if args.workers:
        loader["prefetch_factor"] = 2
    name = "query_adapter_images"
    config = {"base": {"seed": 1234}, "test": {"checkpoint": [str(args.checkpoint)]},
              "data_test": {name: {"name": name, "data_type": "image_classification",
                  "data_path": str(input_csv), "n_class": len(classes), "normalize": None}},
              "dataloader": {"test": loader}}
    evaluator = Evaluator(config, [str(args.checkpoint)], "test")
    if evaluator.device.type != args.device:
        raise RuntimeError("Evaluator device differs from requested device")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ignored = _strict_load_published_checkpoint(evaluator.model, checkpoint["model"])
    del checkpoint
    evaluator.model.requires_grad_(False).eval()
    if evaluator.model.num_radzero_heads != metadata["num_heads"]:
        raise ValueError("Saved adapter FLaS head count differs from the model")
    current_original, current_prompts = original_queries_from_evaluator(evaluator, classes)
    if current_prompts != metadata["prompts"]:
        raise ValueError("Original prompt text/order changed")
    np.testing.assert_allclose(current_original, original, atol=2e-5, rtol=1e-5)
    eta = float(metadata["eta"])
    matcher = QueryRefinementMatcher(evaluator.model, np.stack((original, refined)), (0., eta),
        use_all_image_tokens=evaluator.use_all_image_tokens)
    branches = [(1, "refined")]
    if args.include_baseline:
        branches.insert(0, (0, "baseline"))
    record = {"adapter": str(adapter), "adapter_sha256": sha256(adapter),
              "metadata_sha256": sha256(metadata_path), "checkpoint_sha256": CHECKPOINT_SHA,
              "class_names": classes, "prompts": current_prompts, "eta": eta,
              "images": len(paths), "include_baseline": args.include_baseline,
              "batch_size": args.batch_size, "device": str(evaluator.device),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "no_fitting": True, "no_network_weight_updates": True,
              "query_validation_max_abs_error": float(np.max(np.abs(current_original - original))),
              "ignored_checkpoint_keys": ignored,
              "map_semantics": "grounding_map is raw scaled cosine with CLS when attended; similarity_scores is its spatial patch grid, without CLS. Neither is a segmentation mask.",
              "source_sha256": {str(Path(__file__).resolve()): sha256(__file__)}}
    write_json(args.output / "run_config.json", record)
    write_json(args.output / "image_paths.json", paths)
    write_json(args.output / "class_names.json", classes)
    arrays, offset = {}, 0
    with (args.output / "predictions.csv").open("w", newline="") as handle, torch.inference_mode():
        writer = csv.writer(handle)
        writer.writerow(["index", "image", *[f"{label}:{kind}:{c}" for _, label in branches
                          for kind in ("logit", "probability") for c in classes]])
        for batch in evaluator.data_loader_dict[name]:
            stop = offset + len(batch["images"])
            if batch["image_paths"] != paths[offset:stop]:
                raise ValueError("Input image order changed")
            vision = matcher.encode_vision(batch["images"])
            batch_values = {}
            for index, label in branches:
                result = matcher.match_prepared(vision, matcher.queries[index], return_maps=True)
                values = {key: result[key].cpu().numpy()
                          for key in ("logits", "grounding_map", "similarity_scores")}
                values["probabilities"] = torch.sigmoid(result["logits"]).cpu().numpy()
                batch_values[label] = values
                for key, value in values.items():
                    if not np.isfinite(value).all():
                        raise ValueError(f"Nonfinite {label} {key}")
                    if (label, key) not in arrays:
                        arrays[label, key] = np.lib.format.open_memmap(
                            args.output / f"{label}_{key}.npy", mode="w+", dtype=np.float32,
                            shape=(len(paths), *value.shape[1:]))
                    arrays[label, key][offset:stop] = value
            for index, path in enumerate(paths[offset:stop]):
                writer.writerow([offset + index, path, *[float(value)
                    for _, label in branches for key in ("logits", "probabilities")
                    for value in batch_values[label][key][index]]])
            offset = stop
            write_json(args.output / "progress.json", {"status": "inference",
                "images_completed": offset, "images_total": len(paths)})
            print(f"{metadata['dataset']}: {offset}/{len(paths)}", flush=True)
    if offset != len(paths):
        raise ValueError("Incomplete inference")
    for array in arrays.values():
        array.flush()
    if any(parameter.requires_grad for parameter in evaluator.model.parameters()):
        raise ValueError("Unexpected trainable model parameter")
    result = {**record, "status": "complete", "elapsed_seconds": time.time() - start,
              "output_arrays": {f"{label}_{key}.npy": list(value.shape)
                                for (label, key), value in arrays.items()}}
    write_json(args.output / "result.json", result)
    write_json(args.output / "progress.json", {"status": "complete", "images_completed": offset})
    print(f"Saved frozen-query predictions to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="query_adapter.npz or its directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", nargs="+", help="One or more prepared PNG/JPG paths")
    source.add_argument("--image-list", type=Path, help="One path per line, relative to this list")
    source.add_argument("--manifest", type=Path, help="CSV with image column, relative to this CSV")
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-baseline", action="store_true", help="Also save the original query outputs")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or workers")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write_json(args.output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
