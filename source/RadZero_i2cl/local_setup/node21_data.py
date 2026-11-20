"""Real Node21 public development images for RadZero zero-shot classification.

Only ``metadata.csv`` is used. ``simulated_metadata.csv`` contains artificial
annotations for the challenge generation task and is not a source of targets.
The public development collection is not the hidden Node21 challenge test set.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ROOT = Path(
    "/nvme1/zjx/code/medical/alpharad-assets/data_images/Node21"
)
PROMPT_FILE = (
    Path(__file__).resolve().parents[2]
    / "radzero-assets/model/data/ChestXray14/chestxray14_test_text.json"
)
PREPROCESSING_PROTOCOL = (
    "Read the released processed MHA without orientation changes, extra crop, "
    "or flips; accept a scalar 2D image or a scalar 3D image with one z slice; "
    "cv2.normalize over the complete image to uint8 [0,255]; convert to RGB; "
    "apply the unmodified official RadZero image processor (518x518)."
)


def _integer(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description}: expected an integer, got {value!r}") from exc
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{description}: expected a finite integer, got {value!r}")
    return int(number)


def _mha_image_hw(path):
    """Read dimensions without decompressing pixels or altering orientation."""
    fields = {}
    with path.open("rb") as handle:
        for _ in range(128):
            line = handle.readline(8192)
            if not line:
                break
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"Invalid MHA text header: {path}") from exc
            if "=" not in text:
                continue
            key, value = (part.strip() for part in text.split("=", 1))
            fields[key] = value
            if key == "ElementDataFile":
                break
        else:
            raise ValueError(f"Unreasonably long MHA header: {path}")
    if fields.get("ElementDataFile") != "LOCAL":
        raise ValueError(f"Expected a self-contained MHA image: {path}")
    dimension = _integer(fields.get("NDims"), f"{path} NDims")
    sizes = [
        _integer(value, f"{path} DimSize")
        for value in fields.get("DimSize", "").split()
    ]
    components = _integer(fields.get("ElementNumberOfChannels", "1"), f"{path} channels")
    if (
        components != 1
        or dimension not in (2, 3)
        or len(sizes) != dimension
        or min(sizes, default=0) <= 0
        or (dimension == 3 and sizes[2] != 1)
    ):
        raise ValueError(f"Expected scalar 2D or single-z-slice MHA image: {path}, {fields}")
    return sizes[1], sizes[0]


def load_node21(root=DEFAULT_ROOT, limit=0):
    """Return one image-level target per unique real image, sorted by filename.

    Paths are absolute strings. ``boxes`` is a list of per-image ``[x,y,w,h]``
    lists in released image coordinates; it is retained for auditing, not used
    for image-level classification. All metadata rows and image headers are
    validated before applying a positive ``limit`` for a smoke run.
    """
    root = Path(root).expanduser().resolve()
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    processed = root if (root / "metadata.csv").is_file() else root / "cxr_images/proccessed_data"
    table = processed / "metadata.csv"
    image_root = processed / "images"
    with table.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"img_name", "label", "x", "y", "width", "height"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required metadata columns in {table}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata: {table}")
    grouped = {}
    for row_number, row in enumerate(rows, start=2):
        filename = row["img_name"].strip()
        if (
            not filename
            or Path(filename).name != filename
            or Path(filename).suffix.casefold() != ".mha"
            or "\\" in filename
        ):
            raise ValueError(f"Unsafe or invalid image filename on CSV line {row_number}: {filename!r}")
        label = _integer(row["label"], f"{filename} label")
        if label not in (0, 1):
            raise ValueError(f"Nonbinary label for {filename}: {label}")
        record = grouped.setdefault(filename, {"label": label, "boxes": []})
        if record["label"] != label:
            raise ValueError(f"Conflicting labels for repeated image: {filename}")
        if label:
            box = [
                _integer(row[key], f"{filename} {key}")
                for key in ("x", "y", "width", "height")
            ]
            if box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f"Invalid positive bounding box for {filename}: {box}")
            record["boxes"].append(box)
        else:
            for key in ("x", "y", "width", "height"):
                value = row[key].strip().casefold()
                if value not in ("", "nan", "none", "null") and _integer(value, f"{filename} {key}") != 0:
                    raise ValueError(f"Negative image has a nonzero bounding box: {filename}, {row}")

    names = sorted(grouped)
    paths, labels, boxes, sizes = [], [], [], []
    for filename in names:
        path = image_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Node21 image: {path}")
        height, width = _mha_image_hw(path)
        record = grouped[filename]
        if bool(record["boxes"]) != bool(record["label"]):
            raise ValueError(f"Label and real bounding box presence disagree: {filename}")
        for x, y, box_width, box_height in record["boxes"]:
            if x + box_width > width or y + box_height > height:
                raise ValueError(
                    f"Bounding box outside {height}x{width} image: {filename}, "
                    f"{[x, y, box_width, box_height]}"
                )
        paths.append(str(path.resolve()))
        labels.append([record["label"]])
        boxes.append(record["boxes"])
        sizes.append([height, width])

    prompt_entries = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    prompt = prompt_entries["5"][0]
    if prompt != "There is Lung Nodule":
        raise ValueError(f"The official ChestXray14 Nodule prompt changed: {prompt!r}")
    full_count = len(paths)
    full_positives = sum(label[0] for label in labels)
    full_boxes = sum(len(items) for items in boxes)
    selected = min(limit, full_count) if limit else full_count
    paths, labels, boxes, sizes, names = (
        value[:selected] for value in (paths, labels, boxes, sizes, names)
    )
    labels = np.asarray(labels, dtype=np.int8)
    metadata = {
        "dataset": "Node21",
        "root": str(root),
        "metadata_csv": str(table),
        "split": "entire public processed development/training collection; not hidden challenge test",
        "evaluation_unit": "image",
        "task": "zero-shot nodule presence classification",
        "image_order": "lexicographic ascending img_name; group repeated rows by img_name",
        "label_encoding": "0=without nodule, 1=with at least one nodule",
        "label_source": "real metadata.csv label, checked against real bounding box presence",
        "box_encoding": "[x,y,width,height] in released processed image coordinates",
        "simulated_metadata_used": False,
        "source_rows": len(rows),
        "full_unique_images": full_count,
        "full_positive_images": full_positives,
        "full_negative_images": full_count - full_positives,
        "full_box_rows": full_boxes,
        "unique_images": selected,
        "positive_images_per_class": {"Nodule": int(labels.sum())},
        "negative_images": selected - int(labels.sum()),
        "box_rows": sum(len(items) for items in boxes),
        "path_label_and_box_bounds_validated": True,
        "image_headers_checked": full_count,
        "all_processed_mha_files": len(list(image_root.glob("*.mha"))),
        "unreferenced_mha_files": len(
            {path.name for path in image_root.glob("*.mha")} - set(grouped)
        ),
        "pixel_finiteness_check": "performed for every decoded image in Node21Images",
        "preprocessing": PREPROCESSING_PROTOCOL,
        "prompt_source": "official ChestXray14 JSON key 5 (Lung Nodule)",
        "limit": int(limit),
    }
    return {
        "name": "Node21",
        "class_names": ["Nodule"],
        "prompts": [prompt],
        "image_paths": paths,
        "labels": labels,
        "ids": [Path(name).stem for name in names],
        "boxes": boxes,
        "image_hw": sizes,
        "source_files": [str(table), str(PROMPT_FILE)],
        "metadata": metadata,
    }


def read_node21_pixels(path):
    """Decode released grayscale pixels in their stored array orientation."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "Node21 MHA decoding requires SimpleITK in the RadZero virtual environment"
        ) from exc
    image = sitk.ReadImage(str(path))
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Expected scalar MHA pixels: {path}")
    pixels = sitk.GetArrayFromImage(image)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or min(pixels.shape) <= 0:
        raise ValueError(f"Expected 2D or single-z-slice MHA pixels: {path}, {pixels.shape}")
    if not np.isfinite(pixels).all():
        raise ValueError(f"Nonfinite MHA pixel intensities: {path}")
    if pixels.shape != _mha_image_hw(Path(path)):
        raise ValueError(f"Decoded pixels and MHA header dimensions disagree: {path}")
    return pixels


class Node21Images(Dataset):
    """MHA image adapter with the same output signature as infer_batch.TestImages."""

    def __init__(self, paths, processor):
        self.paths = list(paths)
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        import cv2

        cv2.setNumThreads(0)
        pixels = read_node21_pixels(self.paths[index])
        image_hw = torch.tensor(pixels.shape, dtype=torch.int32)
        normalized = cv2.normalize(
            pixels, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        image = Image.fromarray(normalized).convert("RGB")
        tensor = self.processor(image, return_tensors="pt")["pixel_values"][0]
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite processed tensor: {self.paths[index]}")
        return tensor, image_hw, index
