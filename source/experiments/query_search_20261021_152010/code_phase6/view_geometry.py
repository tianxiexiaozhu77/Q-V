"""Fixed image views and patch-center inverse warping for frozen CXR models.

Images already have the author's standard 518x518 preprocessing/normalization.
Cropping therefore operates on that tensor, not on the original image file.
Raw similarity maps are never softmaxed. Each view's patch map is sampled back
at original-image patch centers, with align_corners=False and geometric coverage
masks. Border interpolation extends its nearest patch sample inside the crop;
outside the crop it contributes neither score nor denominator.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch.nn import functional as F


VERSION = "fixed-multiview-patch-centers-v1"
IMAGE_SIZE = 518
PATCH_GRID = 37
CROP_SIZE = int(IMAGE_SIZE*.8)  # 414, with exact integer bounds recorded below.
_END = IMAGE_SIZE-CROP_SIZE
_CENTER = _END//2
VIEW_SPECS = {
    "identity": {"operation": "identity", "bounds_xyxy": [0, 0, IMAGE_SIZE, IMAGE_SIZE]},
    "hflip": {"operation": "horizontal_flip", "bounds_xyxy": [0, 0, IMAGE_SIZE, IMAGE_SIZE]},
    "center80": {"operation": "crop_resize", "bounds_xyxy": [_CENTER, _CENTER, _CENTER+CROP_SIZE, _CENTER+CROP_SIZE]},
    "tl80": {"operation": "crop_resize", "bounds_xyxy": [0, 0, CROP_SIZE, CROP_SIZE]},
    "tr80": {"operation": "crop_resize", "bounds_xyxy": [_END, 0, IMAGE_SIZE, CROP_SIZE]},
    "bl80": {"operation": "crop_resize", "bounds_xyxy": [0, _END, CROP_SIZE, IMAGE_SIZE]},
    "br80": {"operation": "crop_resize", "bounds_xyxy": [_END, _END, IMAGE_SIZE, IMAGE_SIZE]},
}
for _spec in VIEW_SPECS.values():
    _spec.update({"input_hw": [IMAGE_SIZE, IMAGE_SIZE], "output_hw": [IMAGE_SIZE, IMAGE_SIZE],
                  "resize_mode": "bilinear", "align_corners": False,
                  "requested_crop_fraction": .8 if _spec["operation"] == "crop_resize" else 1.})


def view_spec(name):
    """Return a copy of the fully specified deterministic view metadata."""
    if name not in VIEW_SPECS:
        raise ValueError(f"Unknown view {name!r}")
    return deepcopy(VIEW_SPECS[name])


def apply_view(pixels, view_name):
    """Apply a view to already-preprocessed [B,3,518,518] floating tensors.

    The identity branch returns the identical tensor object without operations.
    Crops use recorded integer slices followed by bilinear resize, with no
    renormalization, antialiasing, or augmentation randomness.
    """
    spec = view_spec(view_name)
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("Expected preprocessed image tensors [B,3,518,518]")
    if not pixels.is_floating_point():
        raise ValueError("Preprocessed image tensors must be floating point")
    if spec["operation"] == "identity":
        return pixels
    if spec["operation"] == "horizontal_flip":
        return pixels.flip(-1)
    left, top, right, bottom = spec["bounds_xyxy"]
    return F.interpolate(pixels[..., top:bottom, left:right], size=(IMAGE_SIZE, IMAGE_SIZE),
                         mode="bilinear", align_corners=False, antialias=False)


def inverse_map(patch_maps, view_name):
    """Return (original-frame maps [...,37,37], bool coverage [37,37]).

    Coordinates use pixel-edge units: original patch centers are (j+.5)*518/37.
    A crop's sample coordinates are 2*(center-left)/crop_width-1, as required
    by grid_sample(...,align_corners=False). Crop validity is based on geometric
    crop bounds rather than the range of patch-center samples.
    """
    spec = view_spec(view_name)
    if not isinstance(patch_maps, torch.Tensor) or patch_maps.ndim < 2 or tuple(patch_maps.shape[-2:]) != (PATCH_GRID, PATCH_GRID):
        raise ValueError("Expected floating patch maps [...,37,37]")
    if not patch_maps.is_floating_point():
        raise ValueError("Patch maps must be floating point")
    valid = torch.ones((PATCH_GRID, PATCH_GRID), dtype=torch.bool, device=patch_maps.device)
    if spec["operation"] == "identity":
        return patch_maps, valid
    if spec["operation"] == "horizontal_flip":
        return patch_maps.flip(-1), valid
    left, top, right, bottom = spec["bounds_xyxy"]
    # FP32 geometry also makes CPU half/bfloat16 input well defined; output is
    # cast back only after interpolation. Native FP32 maps stay FP32 throughout.
    work_dtype = torch.float64 if patch_maps.dtype == torch.float64 else torch.float32
    centers = (torch.arange(PATCH_GRID, device=patch_maps.device, dtype=work_dtype)+.5)*(IMAGE_SIZE/PATCH_GRID)
    gx = 2*(centers-left)/(right-left)-1
    gy = 2*(centers-top)/(bottom-top)-1
    mesh_y, mesh_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([mesh_x, mesh_y], dim=-1)[None]
    valid = ((centers >= top) & (centers < bottom))[:, None] & ((centers >= left) & (centers < right))[None, :]
    values = patch_maps.reshape(-1, 1, PATCH_GRID, PATCH_GRID).to(dtype=work_dtype)
    restored = F.grid_sample(values, grid.expand(len(values), -1, -1, -1),
                             mode="bilinear", padding_mode="border", align_corners=False)
    restored = restored.reshape(patch_maps.shape).to(dtype=patch_maps.dtype)
    return restored*valid.to(dtype=restored.dtype), valid


def _average_maps(maps_by_view, view_names):
    restored, coverage = [], []
    reference = maps_by_view[view_names[0]]
    for name in view_names:
        value = maps_by_view[name]
        if value.shape != reference.shape or value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError("View maps must have the same shape, dtype, and device")
        warped, valid = inverse_map(value, name)
        restored.append(warped)
        coverage.append(valid)
    counts = torch.stack(coverage).sum(0)
    # All declared plans contain identity. Reject incomplete generic plans that
    # leave pixels unsupported instead of inventing map values there.
    if "identity" not in view_names and "hflip" not in view_names and not bool((counts > 0).all()):
        raise ValueError("Requested views do not cover the complete original patch grid")
    return torch.stack(restored).sum(0)/counts.to(dtype=reference.dtype)


def aggregate_outputs(outputs_by_view, view_names):
    """Aggregate native logits and optional raw maps into original coordinates.

    ``outputs_by_view[name]`` contains logits [B,C], optionally similarity_scores
    [B,C,37,37], and optionally grounding_map [B,C,1370] (CLS + 37*37). Keys must
    agree across views. Logits and CLS use arithmetic means. Spatial maps use
    inverse warping and per-patch valid-coverage means, without map softmax or
    min/max normalization. Logits-only classification batches are supported.
    Exactly ["identity"] returns the original output dict unchanged, so controls
    preserve their native bit pattern and even their tensor identity.
    """
    names = list(view_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Require one or more distinct view names")
    for name in names:
        view_spec(name)
        if name not in outputs_by_view:
            raise ValueError(f"Missing native output for view {name!r}")
    outputs = [outputs_by_view[name] for name in names]
    if not all("logits" in output for output in outputs):
        raise ValueError("Each view must provide native logits")
    if names == ["identity"]:
        return outputs[0]
    reference = outputs[0]["logits"]
    if reference.ndim != 2 or any(output["logits"].shape != reference.shape for output in outputs):
        raise ValueError("Native logits must have consistent [B,C] shapes")
    result = {"logits": torch.stack([output["logits"] for output in outputs]).mean(0)}
    for key in ("similarity_scores", "grounding_map"):
        present = [key in output for output in outputs]
        if any(present) and not all(present):
            raise ValueError(f"Native views inconsistently provide {key}")
        if not any(present):
            continue
        if key == "similarity_scores":
            if any(tuple(output[key].shape) != (*reference.shape, PATCH_GRID, PATCH_GRID) for output in outputs):
                raise ValueError("similarity_scores must have shape [B,C,37,37]")
            result[key] = _average_maps({name: outputs_by_view[name][key] for name in names}, names)
        else:
            if any(tuple(output[key].shape) != (*reference.shape, PATCH_GRID*PATCH_GRID+1) for output in outputs):
                raise ValueError("grounding_map must have shape [B,C,1370], including CLS")
            cls = torch.stack([output[key][..., :1] for output in outputs]).mean(0)
            spatial = {name: outputs_by_view[name][key][..., 1:].reshape(*reference.shape, PATCH_GRID, PATCH_GRID)
                       for name in names}
            averaged = _average_maps(spatial, names)
            result[key] = torch.cat([cls, averaged.flatten(-2)], dim=-1)
    return result
