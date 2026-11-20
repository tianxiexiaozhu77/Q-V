"""Offline AlphaRAD inference using the published checkpoint and source methods.

Outputs one sigmoid score per requested finding and grounding maps over the
518x518 model input crop. Scores are not calibrated clinical probabilities.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("ALPHARAD_ASSETS", ROOT / "local_assets"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ASSETS / ".cache/matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT.parent / "data/alpharad/ChestXDet10/test_data/36199.png")
    parser.add_argument("--finding", action="append", help="Finding name, e.g. 'pleural effusion'; repeat for multiple findings.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (select GPU using CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "model/xraydinov2_mpnet/eccv_mpnet_518.tar")
    parser.add_argument("--output", type=Path, default=ROOT / "local_setup/outputs/inference")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import transformers
    from cxrclip.base_evaluator import BaseEvaluator
    from cxrclip.data.data_utils import load_tokenizer, load_transform, transform_image
    from cxrclip.model import build_model

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or run where the selected GPU is accessible.")
    start = time.time()
    # This is the author's downloaded and hash-verified checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    text_dir = str(ASSETS / "pretrained/sentence-transformers/all-mpnet-base-v2")
    image_dir = str(ASSETS / "pretrained/StanfordAIMI/dinov2-base-xray-224")
    config["model"]["text_encoder"]["cache_dir"] = text_dir
    config["model"]["image_encoder"]["cache_dir"] = image_dir
    config["tokenizer"]["cache_dir"] = text_dir
    tokenizer = load_tokenizer(**config["tokenizer"])
    model = build_model(config["model"], config["loss"], tokenizer=tokenizer)
    # The released checkpoint contains this extra scalar, which is absent from
    # the published inference model. Fail on every other missing/extra key.
    state_dict = checkpoint["model"].copy()
    ignored_checkpoint_keys = []
    if "logit_scale_aux" in state_dict and "logit_scale_aux" not in model.state_dict():
        state_dict.pop("logit_scale_aux")
        ignored_checkpoint_keys.append("logit_scale_aux")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    del checkpoint
    model.to(device).eval()
    evaluator = BaseEvaluator(config)
    evaluator.device = device
    evaluator.model = model
    evaluator.datamodule = SimpleNamespace(tokenizer=tokenizer)

    image = Image.open(args.image).convert("RGB")
    pixels = transform_image(load_transform("test", config["transform"]), image, config["image_model"]["normalize"])
    findings = args.finding or ["pleural effusion", "pneumothorax"]
    with torch.inference_mode():
        features = evaluator.encode_image(pixels.unsqueeze(0))
        helper_args = (features["cls_token_raw"], features["cls_token"], features["patch_tokens_raw"], features["patch_tokens"], [], findings)
        predictions, _ = evaluator._zeroshot_binary_dqn_helper(*helper_args)
        attention, per_head = evaluator._zeroshot_pointinggame_dqn_helper(*helper_args)
    assert np.isfinite(predictions).all() and np.isfinite(attention).all()
    spatial = attention[..., 1:] if model.radzero_head_enable_cross_attend_cls_token else attention
    grid_size = int(spatial.shape[-1] ** 0.5)
    assert grid_size ** 2 == spatial.shape[-1]
    input_size = int(pixels.shape[-1])
    heatmaps = F.interpolate(torch.from_numpy(spatial).reshape(1, len(findings), grid_size, grid_size), size=(input_size, input_size), mode="bilinear", align_corners=False).squeeze(0).numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "grounding_maps.npy", heatmaps)
    np.save(args.output / "grounding_per_head.npy", per_head)
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "image": str(args.image.resolve()),
        "device": str(device), "torch": torch.__version__, "transformers": transformers.__version__,
        "input_shape": list(pixels.shape), "grounding_shape": list(heatmaps.shape),
        "raw_attention_shape": list(attention.shape), "per_head_shape": list(per_head.shape),
        "scores": {name: float(predictions[0, i, 0]) for i, name in enumerate(findings)},
        "score_meaning": "Sigmoid alignment score, not a calibrated clinical probability.",
        "map_coordinates": "518x518 resized and center-cropped model input, not original image coordinates.",
        "strict_loaded_after_filter": True,
        "ignored_checkpoint_keys": ignored_checkpoint_keys,
        "elapsed_seconds": time.time() - start,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    rgb = np.clip(pixels.permute(1, 2, 0).numpy() * stds + means, 0, 1)
    fig, axes = plt.subplots(1, len(findings), figsize=(5 * len(findings), 5), squeeze=False)
    for index, name in enumerate(findings):
        ax = axes[0, index]
        ax.imshow(rgb)
        heat = heatmaps[index]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{name}\nscore={result['scores'][name]:.4f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "heatmap.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
