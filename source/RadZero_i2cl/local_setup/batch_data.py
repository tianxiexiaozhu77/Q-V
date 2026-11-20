"""Read the five available RadZero evaluation splits in the authors' order.

Only metadata and file existence are checked here; images are decoded by the
batch runner.  Labels follow the released evaluation code, including OpenI's
image-level substring matching and SIIM's multiple masks per image.
"""

import csv
import json
from pathlib import Path

import numpy as np


DATASETS = ("ChestXDet10", "ChestXray14", "OpenI", "SIIM", "RSNA")

CHESTXDET_CLASSES = [
    "Atelectasis", "Calcification", "Consolidation", "Effusion", "Emphysema",
    "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax",
]
CHESTXRAY_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Lung Mass",
    "Lung Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia",
]
OPENI_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "Fracture", "Opacity", "Lesion",
    "Calcified Granuloma", "Granuloma",
]


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prompts(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {str(index) for index in range(len(data))}:
        raise ValueError(f"Prompt IDs are not consecutive zero-based integers: {path}")
    return [data[str(index)][0] for index in range(len(data))]


def _same_order(actual, expected, description):
    if len(actual) != len(expected):
        raise ValueError(f"{description}: {len(actual)} images but {len(expected)} labels")
    for index, (left, right) in enumerate(zip(actual, expected)):
        if left != right:
            raise ValueError(f"{description}: row {index} differs: {left!r} != {right!r}")


def load_dataset(name, data_root):
    """Return relative image paths, binary labels, prompts and source metadata.

    SIIM uses the released test split (1,704 unique images), not all downloaded
    train/validation/test images.  This loader supplies image-level targets;
    full boxes and masks remain in the recorded original source files.
    """
    aliases = {item.casefold(): item for item in DATASETS}
    if name.casefold() not in aliases:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    name = aliases[name.casefold()]
    root = Path(data_root).expanduser().resolve()
    metadata = {
        "dataset": name,
        "data_root": str(root),
        "evaluation_unit": "image",
        "image_order": "released CSV order; first occurrence for repeated SIIM ImageId",
        "label_encoding": "0=absent, 1=present",
    }
    source_files = []

    if name in ("ChestXDet10", "ChestXray14", "OpenI"):
        stem = {
            "ChestXDet10": "chestXDet10_test",
            "ChestXray14": "chestxray14_test",
            "OpenI": "openi_multi_label",
        }[name]
        image_file = root / name / f"{stem}_image.csv"
        prompt_file = root / name / f"{stem}_text.json"
        rows = _csv(image_file)
        image_paths = [row["Path"] for row in rows]
        prompts = _prompts(prompt_file)
        source_files.extend([image_file, prompt_file])
        metadata.update({
            "source_rows": len(rows),
            "split": "released evaluation split",
            "preprocessing": "cv2.normalize to uint8 0..255, then released image processor",
        })

        if name == "ChestXDet10":
            label_file = root / name / "test.json"
            annotations = json.loads(label_file.read_text(encoding="utf-8"))
            _same_order(
                [Path(path).name for path in image_paths],
                [entry["file_name"] for entry in annotations],
                "ChestXDet10 image/label order",
            )
            class_names = list(CHESTXDET_CLASSES)
            unknown = set().union(*(set(entry["syms"]) for entry in annotations)) - set(class_names)
            if unknown:
                raise ValueError(f"Unexpected ChestXDet10 classes: {sorted(unknown)}")
            labels = [[int(label in entry["syms"]) for label in class_names] for entry in annotations]
            metadata["label_source"] = "test.json syms; multiple boxes for a finding give one positive label"
        elif name == "ChestXray14":
            label_file = root / name / "test_list.txt"
            annotations = [line.split() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            _same_order(
                image_paths,
                [f"ChestXray14/{entry[0]}" for entry in annotations],
                "ChestXray14 image/label order",
            )
            class_names = list(CHESTXRAY_CLASSES)
            if any(len(entry) != len(class_names) + 1 for entry in annotations):
                raise ValueError("ChestXray14 test_list.txt must have path plus 14 binary labels per row")
            labels = [[int(value) for value in entry[1:]] for entry in annotations]
            metadata["label_source"] = "test_list.txt; fixed 14-column order from triple_Chexpert14_result"
        else:
            label_file = root / name / "custom.csv"
            annotations = _csv(label_file)
            _same_order(
                [Path(path).name for path in image_paths],
                [entry["file_name"] for entry in annotations],
                "OpenI image/label order",
            )
            class_names = list(OPENI_CLASSES)
            synonyms = {
                "Pleural_Thickening": ["pleural thickening"],
                "Infiltration": ["infiltrate"],
                "Atelectasis": ["atelectases"],
            }
            labels = []
            for entry in annotations:
                # Match the released str.contains behavior.  In particular,
                # "calcified granuloma" also matches the broader "granuloma".
                text = entry["labels_automatic"] or "-1"
                labels.append([
                    int(any(term in text for term in [label.lower()] + synonyms.get(label, [])))
                    for label in class_names
                ])
            metadata.update({
                "label_source": "custom.csv labels_automatic substring matching, as tripple_openi_rusult_merge",
                "study_ids": [entry["uid"] for entry in annotations],
                "study_aggregation": "none; the released evaluator scores each image independently",
                "study_count": len({entry["uid"] for entry in annotations}),
                "no_finding_class": "not included in the released 18-class evaluation",
            })
        source_files.append(label_file)

    elif name == "SIIM":
        table = root / name / "test.csv"
        rows = _csv(table)
        grouped = {}
        for row in rows:
            grouped.setdefault(row["ImageId"], []).append(row)
        image_paths, labels = [], []
        for image_id, group in grouped.items():
            paths = {row["dicom_path"] for row in group}
            if len(paths) != 1:
                raise ValueError(f"SIIM {image_id} maps to multiple image paths")
            path = group[0]["dicom_path"]
            if Path(path).stem != image_id:
                raise ValueError(f"SIIM ImageId does not match its DICOM filename: {image_id}")
            mask_labels = [int(row[" EncodedPixels"].strip() != "-1") for row in group]
            csv_labels = [int(float(row["Label"])) for row in group]
            if mask_labels != csv_labels:
                raise ValueError(f"SIIM mask and class labels disagree for {image_id}")
            image_paths.append(path)
            labels.append([max(mask_labels)])
        class_names = ["Pneumothorax"]
        prompts = ["There is Pneumothorax"]
        source_files.append(table)
        metadata.update({
            "source_rows": len(rows),
            "split": "test.csv (MGCA split used by RadZero)",
            "image_ids": list(grouped),
            "label_source": "any nonnegative EncodedPixels mask within each ImageId group",
            "mask_encoding": "relative-start RLE from the original SIIM CSV; -1 is absent",
            "preprocessing": "official read_from_dicom conversion, then released image processor",
        })

    else:  # RSNA
        table = root / name / "test.csv"
        rows = _csv(table)
        image_paths = [row["img_path"] for row in rows]
        labels = []
        for row in rows:
            label = int(row["classes"])
            has_boxes = bool(row["boxes"].strip()) and row["boxes"].strip().lower() != "nan"
            if label not in (0, 1) or bool(label) != has_boxes:
                raise ValueError(f"RSNA class and box presence disagree for {row['ID']}")
            labels.append([label])
        class_names = ["Pneumonia"]
        prompts = ["There is Pneumonia"]
        source_files.append(table)
        metadata.update({
            "source_rows": len(rows),
            "split": "test.csv (MedKLIP split used by RadZero)",
            "image_ids": [row["ID"] for row in rows],
            "label_source": "classes; checked against presence of the boxes used by the released evaluator",
            "box_encoding": "x;y;width;height, multiple boxes separated by |",
            "preprocessing": "official read_from_dicom conversion, then released image processor",
        })

    labels = np.asarray(labels, dtype=np.int8)
    if labels.shape != (len(image_paths), len(class_names)):
        raise ValueError(f"{name}: labels have shape {labels.shape}, incompatible with images/classes")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name}: labels contain nonbinary values")
    if len(prompts) != len(class_names):
        raise ValueError(f"{name}: {len(prompts)} prompts but {len(class_names)} classes")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError(f"{name}: duplicate image paths remain after split preparation")
    unsafe = [path for path in image_paths if Path(path).is_absolute() or ".." in Path(path).parts]
    if unsafe:
        raise ValueError(f"{name}: expected relative paths within the data root: {unsafe[:3]}")
    missing = [path for path in image_paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} missing images; first examples: {missing[:3]}")
    metadata.update({
        "unique_images": len(image_paths),
        "positive_images_per_class": dict(zip(class_names, labels.sum(axis=0).tolist())),
        "path_and_label_order_validated": True,
    })
    return {
        "name": name,
        "image_paths": image_paths,
        "labels": labels,
        "class_names": class_names,
        "prompts": prompts,
        "source_files": [str(path) for path in source_files],
        "metadata": metadata,
    }
