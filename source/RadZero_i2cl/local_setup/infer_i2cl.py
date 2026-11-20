#!/usr/bin/env python3
"""Reload a small I2CL checkpoint and score held-out images with frozen RadZero."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from infer import ASSETS_DIR, MODEL_DIR, configure_offline

configure_offline()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from i2cl_data import load_spec
from i2cl_model import RadZeroI2CL
from infer_batch import TestImages, checkpoint_identity, load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--save-maps", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output = args.output.expanduser().resolve()
    if args.output.parent != (ASSETS_DIR / "outputs").resolve():
        raise ValueError("Use a new direct child of radzero-assets/outputs")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["model_identity"] != checkpoint_identity():
        raise ValueError("Original RadZero checkpoint differs from the training checkpoint")
    config = checkpoint["run_config"]
    for filename, key in (("config.json", "model_config_sha256"),
                          ("preprocessor_config.json", "preprocessor_sha256")):
        if hashlib.sha256((MODEL_DIR / filename).read_bytes()).hexdigest() != config[key]:
            raise ValueError("Model/processor configuration differs from training: " + filename)
    spec = load_spec(Path(checkpoint["split_dir"]), checkpoint["dataset"], split="test")
    support = load_spec(Path(checkpoint["split_dir"]), checkpoint["dataset"], split="support")
    if checkpoint["support_sha256"] != support["metadata"]["support_sha256"]:
        raise ValueError("Support split differs from the training checkpoint")
    for key in ("split_fingerprint", "root_split_fingerprint"):
        if spec["metadata"][key] != config["split_metadata"][key]:
            raise ValueError("Held-out evaluation split differs from training: " + key)
    if checkpoint["prompts"] != spec["prompts"] or checkpoint["class_names"] != spec["class_names"]:
        raise ValueError("Prompt or class order differs from the checkpoint")
    if len(set(args.indices)) != len(args.indices) or any(
        i < 0 or i >= len(spec["image_paths"]) for i in args.indices
    ):
        raise ValueError("--indices must be distinct valid held-out test indices")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, processor, tokenizer = load_model(torch.device(args.device))
    wrapper = RadZeroI2CL(model, tokenizer, spec["prompts"])
    wrapper.load_adaptation_state(checkpoint["adaptation_state"])
    wrapper.eval()
    dataset = TestImages(spec["image_paths"], ASSETS_DIR / "data", processor,
                         checkpoint["dataset"] in ("SIIM", "RSNA"))
    loader = DataLoader(Subset(dataset, args.indices), batch_size=args.batch_size, shuffle=False)
    chunks, maps, sizes = [], [], []
    with torch.no_grad():
        for pixels, hw, _indices in loader:
            result = wrapper(pixels, return_maps=args.save_maps)
            chunks.append(result["logits"].cpu().numpy())
            sizes.append(hw.numpy())
            if args.save_maps:
                maps.append(result["similarity_scores"].cpu().numpy())
    logits = np.concatenate(chunks)
    args.output.mkdir()
    fields = ["test_index", "image_path"]
    fields += [f"label:{name}" for name in spec["class_names"]]
    fields += [f"logit:{name}" for name in spec["class_names"]]
    with (args.output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, values in zip(args.indices, logits):
            writer.writerow([i, spec["image_paths"][i], *spec["labels"][i].tolist(), *values.tolist()])
    report = {"dataset": checkpoint["dataset"], "test_indices": args.indices,
              "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "support_sha256": checkpoint["support_sha256"],
              "sample_count": len(args.indices), "logits_shape": list(logits.shape)}
    previous = args.checkpoint.parent / "predictions.csv"
    if previous.is_file():
        with previous.open(newline="") as f:
            saved = {row["image_path"]: row for row in csv.DictReader(f)}
        paths = [spec["image_paths"][i] for i in args.indices]
        if all(path in saved for path in paths):
            expected = np.asarray([[float(saved[path][f"logit:{name}"])
                                    for name in spec["class_names"]] for path in paths])
            np.testing.assert_allclose(logits, expected, atol=2e-5, rtol=1e-4)
            report["reload_max_logit_error"] = float(np.max(np.abs(logits - expected)))
            report["reload_passed"] = True
    if args.save_maps:
        arrays = np.concatenate(maps)
        np.savez_compressed(args.output / "similarity_maps.npz",
                            test_indices=np.asarray(args.indices), logits=logits,
                            similarity_scores=arrays, image_hw=np.concatenate(sizes))
        report["maps_shape"] = list(arrays.shape)
        report["map_description"] = "Raw 37x37 similarity scores; interpolate to image_hw before sigmoid."
    write_json(args.output / "result.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
