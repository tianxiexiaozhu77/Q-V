"""Freeze local AlphaRAD evaluation lists without changing source data.

The four complete classification splits reuse RadZero's released reference
labels; this is not a claim that unpublished AlphaRAD CSVs were recovered.
DICOM conversion executes the author's actual function with its default resize.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDICAL = ROOT.parent
ALPHA = MEDICAL / "data/alpharad"
SHARED = MEDICAL / "data/radzero/prepared"
CXD = ["Atelectasis", "Calcification", "Consolidation", "Effusion", "Emphysema",
       "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax"]
NIH = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
       "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
       "Fibrosis", "Pleural Thickening", "Hernia"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def save_csv(path, rows, fields):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_file(path):
    return {"path": str(path), "sha256": sha(path)}


def author_converter():
    # Execute the original function verbatim without importing unrelated MHA
    # dependencies or the model. In particular, preserve cv2.resize's actual
    # default INTER_LINEAR behavior despite the positional LANCZOS constant.
    import cv2
    import numpy as np
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    cv2.setNumThreads(1)
    source = ROOT / "cxrclip/data/dataformat_converter.py"
    module = ast.parse(source.read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == "resize_dicom_and_save")
    namespace = {"cv2": cv2, "np": np, "pydicom": pydicom, "os": os,
                 "apply_modality_lut": apply_modality_lut, "apply_voi_lut": apply_voi_lut}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def check_paths(paths, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        exists = list(pool.map(lambda p: Path(p).is_file(), paths))
    missing = [str(p) for p, good in zip(paths, exists) if not good]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing source images: {missing[:3]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared-images", type=Path,
                        default=ROOT / "local_assets/data_prepared/author_default512")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Verify and retain already published CSVs.")
    args = parser.parse_args()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=args.resume)
    image_root = args.prepared_images.absolute()
    image_root.mkdir(parents=True, exist_ok=True)
    catalog_path = output / "catalog.json"
    catalog = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_root": str(ALPHA), "shared_image_root": str(SHARED),
               "status": "preparing", "datasets": {}, "unavailable": {},
               "source_script": source_file(Path(__file__)),
               "notes": ["No training or query adaptation. Original complete reference splits are used.",
                         "External rsync may be writing source directories; available files are frozen by each CSV.",
                         "Classification CSVs except published author lists are reconstructed from cited reference labels."]}
    if args.resume:
        catalog = json.loads(catalog_path.read_text())
        catalog.setdefault("preparation_source_versions", [catalog["source_script"]]).append(source_file(Path(__file__)))
        if catalog["status"] != "preparing":
            raise ValueError("Only an interrupted preparing catalog may be resumed")
    atomic_json(catalog_path, catalog)

    def publish(key, rows, classes, task, display, expected, scope, sources, notes=()):
        path = output / f"{key}.csv"
        fields = ["image", "class", "label"] + (["boxes"] if task == "grounding" else [])
        if key in catalog["datasets"]:
            if sha(path) != catalog["datasets"][key]["csv_sha256"] or read_csv(path) != rows:
                raise ValueError(f"Previously published CSV changed: {key}")
            print(f"VERIFIED already published {key}: {len(rows)} rows", flush=True)
            return
        save_csv(path, rows, fields)
        catalog["datasets"][key] = {
            "csv_path": str(path), "csv_sha256": sha(path), "class_names": classes,
            "task": task, "display_name": display, "scope": scope,
            "expected_count": expected, "actual_count": len(rows),
            "unique_images": len({r["image"] for r in rows}),
            "notes": list(notes), "sources": [source_file(p) for p in sources],
            "positive_count": [sum(ast.literal_eval(r["label"])[c] for r in rows)
                               for c in range(len(classes))],
        }
        atomic_json(catalog_path, catalog)
        print(f"READY {key}: {len(rows)} rows; {catalog_path}", flush=True)

    # This published AlphaRAD grounding CSV is kept byte-for-byte in all label
    # fields, with only the image path changed and irrelevant columns removed.
    source = ROOT / "configs/data_csv/chestXDet10_gt.csv"
    rows = []
    for row in read_csv(source):
        suffix = row["image"].split("/chestXDet10/", 1)[1]
        rows.append({"image": str(ALPHA / "ChestXDet10" / suffix),
                     **{k: row[k] for k in ("class", "label", "boxes")}})
    check_paths([r["image"] for r in rows], args.workers)
    publish("chestxdet10_gt", rows, CXD, "grounding", "ChestXDet10", 459,
            "complete published AlphaRAD positive-image grounding CSV", [source])

    source = SHARED / "ChestXDet10/test.json"
    image_csv = SHARED / "ChestXDet10/chestXDet10_test_image.csv"
    annotations = json.loads(source.read_text())
    image_names = [Path(r["Path"]).name for r in read_csv(image_csv)]
    assert image_names == [r["file_name"] for r in annotations]
    rows = []
    for row in annotations:
        classes = [c for c in CXD if c in row["syms"]]
        rows.append({"image": str(ALPHA / "ChestXDet10/test_data" / row["file_name"]),
                     "class": repr(classes), "label": repr([int(c in classes) for c in CXD])})
    check_paths([r["image"] for r in rows], args.workers)
    publish("chestxdet10", rows, CXD, "classification", "ChestXDet10", 542,
            "complete released reference test split, including all-negative images", [source, image_csv])

    source = SHARED / "ChestXray14/test_list.txt"
    image_csv = SHARED / "ChestXray14/chestxray14_test_image.csv"
    entries = [line.split() for line in source.read_text().splitlines() if line.strip()]
    assert [r["Path"] for r in read_csv(image_csv)] == ["ChestXray14/" + r[0] for r in entries]
    rows = []
    for entry in entries:
        assert len(entry) == 15
        labels = list(map(int, entry[1:]))
        assert set(labels) <= {0, 1}
        # Shared source is complete and stable; avoid mixing an in-progress
        # duplicate rsync with the already validated released split.
        rows.append({"image": str(SHARED / "ChestXray14" / entry[0]),
                     "class": repr([c for c, y in zip(NIH, labels) if y]), "label": repr(labels)})
    check_paths([r["image"] for r in rows], args.workers)
    publish("chest14", rows, NIH, "classification", "ChestXray14", 22433,
            "complete released reference test split; full 22433 images, no few-shot holdout", [source, image_csv],
            ["AlphaRAD's CSV and class constant are not released; labels follow the fixed NIH 14-column order.",
             "Classes Mass/Nodule are plain disease names; no RadZero description prompts are copied."])

    convert = author_converter()
    converter_file = ROOT / "cxrclip/data/dataformat_converter.py"
    for name, key, classes in (("RSNA", "rsna_pneumonia", ["Pneumonia"]),
                               ("SIIM", "siim_pneumothorax", ["Pneumothorax"])):
        source = SHARED / name / "test.csv"
        source_rows = read_csv(source)
        records = []
        if name == "RSNA":
            for row in source_rows:
                label = int(row["classes"])
                assert label in (0, 1)
                assert bool(label) == bool(row["boxes"].strip())
                records.append((SHARED / row["img_path"], label, row["ID"]))
        else:
            grouped = {}
            for row in source_rows:
                grouped.setdefault(row["ImageId"], []).append(row)
            for image_id, group in grouped.items():
                assert len({r["dicom_path"] for r in group}) == 1
                labels = [int(r[" EncodedPixels"].strip() != "-1") for r in group]
                assert labels == [int(float(r["Label"])) for r in group]
                records.append((SHARED / group[0]["dicom_path"], max(labels), image_id))
        assert len(records) == (5337 if name == "RSNA" else 1704)
        check_paths([r[0] for r in records], args.workers)
        destination = image_root / name
        destination.mkdir(exist_ok=True)

        def process(record):
            import cv2
            path, label, image_id = record
            target = destination / f"{image_id}.png"
            source_hash = sha(path)
            # A separate immutable ledger verifies reused conversion outputs.
            receipt = target.with_suffix(".json")
            if target.exists() or receipt.exists():
                old = json.loads(receipt.read_text())
                if old["source_sha256"] != source_hash or old["png_sha256"] != sha(target):
                    raise ValueError(f"Existing converted image provenance mismatch: {target}")
            else:
                convert(str(path), str(target), do_resize=True)
                pixels = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
                if pixels is None or pixels.ndim != 2 or min(pixels.shape) != 512:
                    raise ValueError(f"Invalid converted image: {target}")
                old = {"source": str(path), "source_sha256": source_hash,
                       "png": str(target), "png_sha256": sha(target),
                       "converter_sha256": sha(converter_file), "do_resize": True,
                       "shape": list(pixels.shape), "actual_resize_interpolation": "INTER_LINEAR"}
                atomic_json(receipt, old)
            return {"image": str(target), "class": repr(classes if label else []),
                    "label": repr([label])}, old

        rows, provenance = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (row, receipt) in enumerate(pool.map(process, records), 1):
                rows.append(row)
                provenance.append(receipt)
                if i % 250 == 0:
                    print(f"CONVERT {name}: {i}/{len(records)}", flush=True)
        conversion_manifest = output / f"{key}_conversion.json"
        atomic_json(conversion_manifest, provenance)
        publish(key, rows, classes, "classification", name, len(records),
                "complete released reference test split; no support images removed", [source, converter_file, conversion_manifest],
                ["Author converter executed unchanged with do_resize=True (short edge 512).",
                 "Modality/VOI LUT, MONOCHROME1 inversion, min-max uint8 normalization preserved.",
                 "Actual cv2.resize interpolation is default INTER_LINEAR because LANCZOS is passed in the dst position.",
                 "SIIM repeated masks grouped by ImageId; positive if any mask present." if name == "SIIM" else
                 "RSNA labels checked against bounding-box presence."])

    # Snapshot counts once; never silently evaluate whatever subset happens to
    # have arrived from an external copy in progress.
    inventory = {}
    for directory in sorted(ALPHA.iterdir()):
        extensions = {}
        for base, dirs, files in os.walk(directory):
            for filename in files:
                ext = Path(filename).suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        inventory[directory.name] = {"path": str(directory), "readable": os.access(directory, os.R_OK | os.X_OK),
                                     "file_extensions": extensions,
                                     "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "inventory.json", inventory)
    for name in ("CovidKaggle", "Montgomery", "ShenzhenXray", "PadChestSample", "CheXpert5x200",
                 "CheXlocalizeSample", "Node21", "QaTaCOV19"):
        catalog["unavailable"][name] = {"status": "not_prepared_in_initial_complete_split_batch",
                                        "inventory": inventory.get(name),
                                        "note": "Inspect current copying status and annotations before further evaluation; no incomplete subset treated as full paper test."}
    catalog["status"] = "complete"
    catalog["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(catalog_path, catalog)
    print(f"COMPLETE {catalog_path}", flush=True)


if __name__ == "__main__":
    main()
