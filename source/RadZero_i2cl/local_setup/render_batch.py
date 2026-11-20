#!/usr/bin/env python3
"""Render an existing RadZero batch prediction on CPU, without loading a model."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from infer import ASSETS_DIR, MODEL_DIR


DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")


def read_evaluation_image(path, dicom):
    """Preserve original dimensions with the released evaluation pixel scaling."""
    import cv2

    cv2.setNumThreads(0)
    if dicom:
        import pydicom

        dcm = pydicom.dcmread(path)
        pixels = dcm.pixel_array
        maximum = float(pixels.max())
        pixels = (
            cv2.convertScaleAbs(pixels, alpha=255.0 / maximum)
            if maximum else np.zeros_like(pixels, dtype=np.uint8)
        )
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            pixels = cv2.bitwise_not(pixels)
    else:
        with Image.open(path) as original:
            pixels = cv2.normalize(
                np.array(original), None, 0, 255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U,
            )
    return Image.fromarray(pixels).convert("RGB")


def load_prediction(directory, index, manifest):
    """Find one saved chunk and match its row to the prediction/source table."""
    selected = None
    for path in sorted((directory / "batches").glob("batch_*.npz")):
        match = re.fullmatch(r"batch_(\d+)_(\d+)\.npz", path.name)
        if match and int(match[1]) <= index < int(match[2]):
            selected = path
            break
    if selected is None:
        raise FileNotFoundError(f"No saved prediction for index {index} under {directory}")
    with np.load(selected, allow_pickle=False) as chunk:
        locations = np.flatnonzero(chunk["indices"] == index)
        if len(locations) != 1:
            raise ValueError(f"Expected one occurrence of index {index} in {selected}")
        row = int(locations[0])
        logits = chunk["logits"][row].copy()
        maps = chunk["similarity_scores"][row].copy()
        image_hw = tuple(int(value) for value in chunk["image_hw"][row])
    count = len(manifest["class_names"])
    if logits.shape != (count,) or maps.shape != (count, 37, 37):
        raise ValueError(f"Invalid saved logits/maps shapes: {logits.shape}, {maps.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(maps).all():
        raise ValueError("Saved prediction contains non-finite values")

    prediction_file = directory / "predictions.csv"
    row_data = None
    if prediction_file.is_file():
        with prediction_file.open(newline="", encoding="utf-8") as handle:
            row_data = next((row for row in csv.DictReader(handle) if int(row["index"]) == index), None)
        if row_data is None:
            raise ValueError(f"Index {index} is absent from {prediction_file}")
        image_path = row_data["image_path"]
        labels = [int(row_data[f"label:{name}"]) for name in manifest["class_names"]]
        csv_logits = [float(row_data[f"logit:{name}"]) for name in manifest["class_names"]]
        if not np.allclose(csv_logits, logits, rtol=0, atol=1e-7):
            raise ValueError("Prediction CSV and saved chunk logits do not agree")
    else:
        # A running dataset has chunks before its final predictions.csv exists.
        # Verify the recorded metadata before reconstructing the original order.
        from batch_data import load_dataset

        for source, digest in manifest["source_sha256"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source metadata changed since inference: {source}")
        spec = load_dataset(manifest["dataset"], manifest["metadata"]["data_root"])
        if spec["class_names"] != manifest["class_names"]:
            raise ValueError("Class order differs from the saved inference manifest")
        image_path = spec["image_paths"][index]
        labels = spec["labels"][index].tolist()
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Expected a relative image path, got {image_path!r}")
    return {
        "logits": logits, "maps": maps, "image_hw": image_hw,
        "image_path": image_path, "labels": labels, "chunk": str(selected),
    }


def render(run_dir, dataset, index=0, class_index=None, output=None):
    directory = Path(run_dir).expanduser().resolve() / dataset
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or not 0 <= index < manifest["image_count"]:
        raise ValueError("Dataset or image index does not match the saved manifest")
    prediction = load_prediction(directory, index, manifest)
    automatically_selected = class_index is None
    if class_index is None:
        class_index = int(np.argmax(prediction["logits"]))
    if not 0 <= class_index < len(manifest["class_names"]):
        raise ValueError(f"Class index must be in 0..{len(manifest['class_names']) - 1}")
    output = Path(output).expanduser().resolve() if output else (
        directory.parent / "preview" / f"{dataset}_{index:06d}_class{class_index:02d}"
    )
    if output == MODEL_DIR or MODEL_DIR in output.parents:
        raise ValueError("Preview outputs must be outside the original checkpoint snapshot")
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(manifest["metadata"]["data_root"])
    image_path = data_root / prediction["image_path"]
    image = read_evaluation_image(image_path, dataset in ("SIIM", "RSNA"))
    if (image.height, image.width) != prediction["image_hw"]:
        raise ValueError("Image dimensions differ from the dimensions saved during inference")

    import torch
    import torch.nn.functional as F

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    raw_map = torch.from_numpy(prediction["maps"][class_index].astype(np.float32))
    with torch.inference_mode():
        resized = F.interpolate(
            raw_map[None, None], size=prediction["image_hw"],
            mode="bilinear", align_corners=False,
        )
        similarity = resized.sigmoid()[0, 0].numpy()
    np.save(output / "similarity.npy", similarity)

    os.environ.setdefault("MPLCONFIGDIR", str(ASSETS_DIR / ".cache/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    colors = (colormaps["turbo"](np.clip(similarity, 0, 1))[..., :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(colors)
    overlay = Image.blend(image, heatmap, alpha=0.45)
    image.save(output / "image.png")
    heatmap.save(output / "heatmap.png")
    overlay.save(output / "overlay.png")
    logit = float(prediction["logits"][class_index])
    score = float(1.0 / (1.0 + np.exp(-np.float64(logit))))
    class_name = manifest["class_names"][class_index]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title(f"{dataset} / image {index}")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{class_name} / matching score {score:.4f}")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="turbo"),
        ax=axes[1], fraction=0.045, pad=0.03, label="Patch similarity (sigmoid)",
    )
    figure.savefig(output / "preview.png", dpi=140)
    plt.close(figure)
    result = {
        "dataset": dataset, "index": index, "class_index": class_index,
        "class_name": class_name, "prompt": manifest["prompts"][class_index],
        "class_selection": "highest matching score" if automatically_selected else "explicit class index",
        "label": int(prediction["labels"][class_index]),
        "logit": logit, "matching_score": score,
        "image": str(image_path), "image_hw": list(prediction["image_hw"]),
        "source_chunk": prediction["chunk"], "device": "cpu", "model_loaded": False,
        "similarity_shape": list(similarity.shape),
        "similarity_min": float(similarity.min()), "similarity_max": float(similarity.max()),
        "map_transform": "bilinear interpolate raw 37x37 scores to original image dimensions, align_corners=False, then sigmoid",
        "image_transform": "Original dimensions; official evaluation PNG minmax / DICOM max scaling and MONOCHROME1 inversion",
        "colormap": "turbo", "color_scale": [0.0, 1.0], "overlay_alpha": 0.45,
        "note": "Image-text and patch similarity scores are not calibrated disease probabilities or binary segmentation masks.",
        "outputs": {name: str(output / name) for name in (
            "image.png", "heatmap.png", "overlay.png", "preview.png", "similarity.npy",
        )},
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--index", type=int, default=0, help="Zero-based index in the released test CSV")
    parser.add_argument("--class-index", type=int, help="Zero-based class index; default selects highest score")
    parser.add_argument("--run-dir", type=Path, default=ASSETS_DIR / "outputs/public5_gpu0")
    parser.add_argument("--output", type=Path, help="Preview output directory")
    args = parser.parse_args()
    if args.index < 0 or (args.class_index is not None and args.class_index < 0):
        parser.error("Image and class indices must be nonnegative")
    result = render(args.run_dir, args.dataset, args.index, args.class_index, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
