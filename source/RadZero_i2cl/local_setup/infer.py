#!/usr/bin/env python3
"""Run the released RadZero checkpoint using only downloaded local assets."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


from asset_paths import ASSETS_DIR, PROJECT_ROOT
MODEL_DIR = ASSETS_DIR / "model"
VISION_DIR = ASSETS_DIR / "pretrained/StanfordAIMI/dinov2-base-xray-224"
TEXT_DIR = ASSETS_DIR / "pretrained/sentence-transformers/all-mpnet-base-v2"


def configure_offline():
    # Set these before importing Transformers or huggingface_hub. The custom
    # checkpoint constructors load the two base encoders internally as well.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ASSETS_DIR / ".cache/huggingface"))
    os.environ.setdefault(
        "HF_MODULES_CACHE", str(ASSETS_DIR / ".cache/huggingface/modules")
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def select_device(requested, torch):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return torch.device(requested)


def find_images(image_path=None, count=1):
    if image_path:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        return [path] * count
    image_root = ASSETS_DIR / "data_images"
    candidates = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    # Prefer the downloaded ChestXDet test set when other image data is present.
    candidates.sort(
        key=lambda path: (
            not ("chestxdet" in str(path).lower() and "test" in str(path).lower()),
            str(path),
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No real X-ray images found under {image_root}. Finish downloading "
            "and extracting the ChestXDet test images, or supply --image /path/to/xray.png."
        )
    return [candidates[index % len(candidates)] for index in range(count)]


def read_local_config():
    for directory in (MODEL_DIR, VISION_DIR, TEXT_DIR):
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model configuration: {directory}")
    with (MODEL_DIR / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["vision_config"]["pretrained_name_or_path"] = str(VISION_DIR)
    config["vision_config"]["pretrained_dir"] = str(ASSETS_DIR / "pretrained")
    config["text_config"]["pretrained_name_or_path"] = str(TEXT_DIR)
    config["text_config"]["pretrained_tokenizer_name_or_path"] = str(TEXT_DIR)
    config["pretrained_dir"] = str(ASSETS_DIR / "pretrained")
    if isinstance(config.get("kwargs"), dict):
        config["kwargs"]["pretrained_dir"] = config["pretrained_dir"]
    return config


def load_hf_utils():
    # Never add the snapshot to sys.path or import a generic module named utils.
    module_path = MODEL_DIR / "utils.py"
    spec = importlib.util.spec_from_file_location("radzero_downloaded_hf_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the downloaded inference helper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="PNG, JPEG, or DICOM X-ray; defaults to a downloaded test image")
    parser.add_argument("--text", default="There is pleural effusion.", help="One English finding prompt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "local_setup/outputs/inference")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")

    configure_offline()
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    image_path = find_images(args.image)[0]
    output_dir = args.output.expanduser().resolve()
    if output_dir == MODEL_DIR or MODEL_DIR in output_dir.parents:
        raise ValueError("Choose an output directory outside the original model snapshot.")
    device = select_device(args.device, torch)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    config_dict = read_local_config()
    config_class = get_class_from_dynamic_module(
        config_dict["auto_map"]["AutoConfig"], str(MODEL_DIR), local_files_only=True
    )
    config = config_class.from_dict(config_dict)
    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    hf_utils = load_hf_utils()
    original_image = hf_utils.load_image(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not decode the input image: {image_path}")
    original_image = original_image.convert("RGB")
    with torch.inference_mode():
        probability, similarity = hf_utils.model_inference(
            original_image, args.text, model, image_processor, tokenizer
        )
    if probability.numel() != 1:
        raise RuntimeError(f"Expected a single image-text score, got {tuple(probability.shape)}")
    if not torch.isfinite(probability).all() or not torch.isfinite(similarity).all():
        raise RuntimeError("Inference produced a non-finite probability or similarity map.")
    similarity_array = similarity.detach().float().cpu().numpy()
    if similarity_array.shape != (original_image.height, original_image.width):
        raise RuntimeError(f"Unexpected similarity map shape: {similarity_array.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "similarity.npy", similarity_array)
    # Use a fixed 0..1 scale so the visualization keeps the official map's values.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    heatmap_rgb = (colormaps["turbo"](np.clip(similarity_array, 0, 1))[..., :3] * 255).astype(np.uint8)
    colored = Image.fromarray(heatmap_rgb)
    Image.blend(original_image, colored, alpha=0.45).save(output_dir / "heatmap.png")

    result = {
        "image": str(image_path),
        "text": args.text,
        "device": str(device),
        "model_directory": str(MODEL_DIR),
        "torch_version": torch.__version__,
        "image_size": {"width": original_image.width, "height": original_image.height},
        "processed_size": image_processor.size,
        "similarity_probability": float(probability.item()),
        "similarity_map_shape": list(similarity_array.shape),
        "similarity_map_min": float(similarity_array.min()),
        "similarity_map_max": float(similarity_array.max()),
        "similarity_map": str(output_dir / "similarity.npy"),
        "heatmap": str(output_dir / "heatmap.png"),
        "heatmap_scale": [0.0, 1.0],
        "note": "The score is the official sigmoid image-text similarity, not a calibrated disease probability.",
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    (output_dir / "result.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
