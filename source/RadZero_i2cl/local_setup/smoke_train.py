#!/usr/bin/env python3
"""Check the source training path with real images, finite losses and gradients."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from local_setup.infer import (
    ASSETS_DIR,
    TEXT_DIR,
    VISION_DIR,
    configure_offline,
    find_images,
    select_device,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional real image reused in both batch entries")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--image-size", type=int, default=518, help="Default is the official 518; use 224 for a smaller smoke test")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "local_setup/outputs/smoke_train.json")
    args = parser.parse_args()
    if args.image_size < 14 or args.image_size % 14:
        parser.error("--image-size must be a positive multiple of the 14-pixel vision patch size")

    configure_offline()
    import torch
    import yaml
    from PIL import Image
    from exp.cxr_pt.model import load_model

    torch.manual_seed(42)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = select_device(args.device, torch)
    image_paths = find_images(args.image, count=2)
    output_path = args.output.expanduser().resolve()
    model_snapshot = ASSETS_DIR / "model"
    if output_path == model_snapshot or model_snapshot in output_path.parents:
        raise ValueError("Choose an output path outside the original model snapshot.")

    config_path = PROJECT_ROOT / "exp/cxr_pt/configs/radzero.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle)["model"])
    model_config = config["model_config"]
    model_config["vision_config"]["pretrained_name_or_path"] = str(VISION_DIR)
    model_config["vision_config"]["img_size"] = args.image_size
    model_config["text_config"]["pretrained_name_or_path"] = str(TEXT_DIR)
    model_config["text_config"]["pretrained_tokenizer_name_or_path"] = str(TEXT_DIR)
    model_config["pretrained_dir"] = str(ASSETS_DIR / "pretrained")
    config["dtype"] = torch.float32
    config["pretrained_ckpt"] = None
    config["adapter_ckpt"] = None
    config["lora_config"]["use_lora"] = False
    models = load_model(config, output_dir=None)
    model = models["model"].to(device)
    tokenizer, processor = models["tokenizer"], models["image_processor"]
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    pixels = processor(images, return_tensors="pt")["pixel_values"].to(device)
    prompts = ["There is pleural effusion.", "There is fibrosis."]
    phrases = [
        tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(device)
        for prompt in prompts
    ]
    class_phrases = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)

    checks = []
    for batch_size in (2, 1):
        model.train()
        model.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixels[:batch_size], encoded_key_phrases=phrases[:batch_size])
        loss = outputs["losses"]["loss"]
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"Batch {batch_size}: expected a finite scalar loss, got {loss}")
        loss.backward()
        grads = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
        if not grads or not torch.stack([torch.isfinite(grad).all() for grad in grads]).all():
            raise RuntimeError(f"Batch {batch_size}: missing gradients or non-finite gradients")
        grad_norm = torch.stack([grad.float().square().sum() for grad in grads]).sum().sqrt()
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Batch {batch_size}: gradient norm is not finite")
        if batch_size == 2 and grad_norm.item() == 0:
            raise RuntimeError("The batch-2 contrastive training check produced only zero gradients")
        if any(parameter.grad is not None for parameter in model.vision_model.parameters()):
            raise RuntimeError("The frozen vision encoder unexpectedly received gradients")
        check = {
            "batch_size": batch_size,
            "loss": float(loss.detach().item()),
            "loss_components": {key: float(value.detach().item()) for key, value in outputs["losses"].items()},
            "gradient_tensor_count": len(grads),
            "gradient_l2_norm": float(grad_norm.item()),
            "gradients_finite": True,
            "vision_encoder_frozen": True,
        }
        del outputs, loss, grads, grad_norm
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode():
            prediction = model.compute_logits(pixels[:batch_size], [class_phrases])
        for name in ("logits", "similarity_scores"):
            if not torch.isfinite(prediction[name]).all():
                raise RuntimeError(f"Batch {batch_size}: {name} contains non-finite values")
        if tuple(prediction["logits"].shape) != (batch_size, len(prompts)):
            raise RuntimeError(f"Unexpected logits shape: {tuple(prediction['logits'].shape)}")
        check.update({
            "logits_shape": list(prediction["logits"].shape),
            "similarity_shape": list(prediction["similarity_scores"].shape),
            "logits_finite": True,
            "similarity_finite": True,
        })
        checks.append(check)
        print(json.dumps(check), flush=True)
        del prediction

    result = {
        "status": "passed",
        "device": str(device),
        "torch_version": torch.__version__,
        "image_paths": [str(path) for path in image_paths],
        "image_size": args.image_size,
        "pixel_values_shape": list(pixels.shape),
        "prompts": prompts,
        "source_config": str(config_path),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "checks": checks,
        "note": "Source training model initialized from local base encoders; example prompts are not ground-truth labels. Forward/backward only, no optimizer step or checkpoint changes. Batch-1 contrastive loss can be zero because there are no negative images.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
