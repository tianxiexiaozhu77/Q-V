"""Prepare complete additional AlphaRAD datasets after the external copy arrives.

Missing author rows are recorded, never silently omitted. Node21 uses its
public real source annotations because the AlphaRAD test CSV is unpublished.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from prepare_storage_data import ALPHA, ROOT, atomic_json, read_csv, save_csv, sha, source_file


def header_hw(path):
    fields = {}
    with Path(path).open("rb") as handle:
        for _ in range(128):
            text = handle.readline(8192).decode("ascii").strip()
            if "=" in text:
                key, value = (v.strip() for v in text.split("=", 1))
                fields[key] = value
            if fields.get("ElementDataFile"):
                break
    sizes = [int(v) for v in fields["DimSize"].split()]
    if fields["ElementDataFile"] != "LOCAL" or len(sizes) != 2:
        raise ValueError(f"Author MHA conversion expects self-contained 2D scalar image: {path}")
    return sizes[1], sizes[0]


def load_mha_converter():
    import cv2
    import numpy as np
    import SimpleITK as sitk
    cv2.setNumThreads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    source = ROOT / "cxrclip/data/dataformat_converter.py"
    module = ast.parse(source.read_text())
    node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "resize_mha_and_save")
    namespace = {"np": np, "cv2": cv2, "sitk": sitk, "os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=["qatacovid_seg", "covidkaggle", "shenzhenxray", "montgomery", "node21"])
    args = parser.parse_args()
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
               "datasets": {}, "unavailable": {}, "source_script": source_file(Path(__file__))}
    catalog_path = args.output / "catalog.json"
    atomic_json(catalog_path, catalog)

    def publish(key, rows, classes, task, display, expected, scope, sources, notes):
        path = args.output / f"{key}.csv"
        save_csv(path, rows, list(rows[0]))
        catalog["datasets"][key] = {"csv_path": str(path), "csv_sha256": sha(path),
                "class_names": classes, "task": task, "display_name": display, "scope": scope,
                "expected_count": expected, "actual_count": len(rows),
                "unique_images": len({r["image"] for r in rows}), "notes": notes,
                "sources": [source_file(p) for p in sources]}
        atomic_json(catalog_path, catalog)
        print(f"READY {key}: {len(rows)} rows; {catalog_path}", flush=True)

    specs = {"qatacovid_seg": ("qatacov19.csv", "QaTaCOV19", "/qatarCovid/", "segmentation", ["covid"], 2113),
             "covidkaggle": ("covidkaggle.csv", "CovidKaggle", "/covid_kaggle_256/", "classification", ["Covid"], 1381)}
    for key in (k for k in args.datasets if k in specs):
        filename, family, marker, task, classes, count = specs[key]
        source = ROOT / "configs/data_csv" / filename
        rows = read_csv(source)
        for row in rows:
            row["image"] = str(ALPHA / family / row["image"].split(marker, 1)[1])
        assert len(rows) == count
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            good = list(pool.map(lambda r: Path(r["image"]).is_file(), rows))
        if not all(good):
            catalog["unavailable"][key] = {"status": "source_copy_incomplete", "expected_rows": count,
                    "present_rows": sum(good), "missing_rows": count - sum(good),
                    "missing_examples": [r["image"] for r, ok in zip(rows, good) if not ok][:3]}
            atomic_json(catalog_path, catalog)
            print(f"PENDING {key}: {sum(good)}/{count} author images", flush=True)
            continue
        notes = ["Author CSV row order, labels and masks retained unchanged; only image paths changed."]
        if key == "covidkaggle":
            notes.append("Original 299x299 public images; author's unreleased covid_kaggle_256 preprocessing unavailable.")
        publish(key, rows, classes, task, family, count, "complete published AlphaRAD CSV", [source], notes)

    for key, family, expected, prefix in (("shenzhenxray", "ShenzhenXray", 662, "CHNCXR"),
                                          ("montgomery", "Montgomery", 138, "MCUCXR")):
        if key not in args.datasets:
            continue
        directory = ALPHA / family
        # Select only the released radiographs, never segmentation masks.
        paths = sorted(directory.rglob("CXR_png/" + prefix + "_*.png"))
        if len(paths) != expected:
            catalog["unavailable"][key] = {"status": "source_copy_incomplete", "expected_images": expected,
                                           "present_images": len(paths)}
            atomic_json(catalog_path, catalog)
            print(f"PENDING {key}: {len(paths)}/{expected} source images", flush=True)
            continue
        rows = []
        for path in paths:
            match = re.fullmatch(prefix + r"_\d+_([01])\.png", path.name)
            if not match:
                raise ValueError(f"Unexpected NLM filename/label: {path}")
            label = int(match.group(1))
            rows.append({"image": str(path), "class": repr(["Tuberculosis"] if label else []), "label": repr([label])})
        source_notes = sorted(p for p in directory.rglob("*") if p.is_file()
                              and ("readme" in p.name.casefold() or p.name == "shenzhen_consensus_roi.csv"))
        publish(key, rows, ["Tuberculosis"], "classification", family, expected,
                "complete public NLM source collection; author AlphaRAD CSV unavailable", source_notes,
                ["Target label is the official source filename suffix: _0=normal, _1=abnormal/tuberculosis.",
                 "No source images removed, no validation/test selection based on model predictions.",
                 f"Use actual official release count {expected}; AlphaRAD paper lists {expected + 1}.",
                 "Tuberculosis disease query is a local reconstruction because author's class constant is unpublished."])

    if "node21" in args.datasets:
        processed = ALPHA / "Node21/cxr_images/proccessed_data"
        source = processed / "metadata.csv"
        groups = {}
        for row in read_csv(source):
            name = row["img_name"]
            if Path(name).name != name or not name.endswith(".mha"):
                raise ValueError(f"Invalid MHA name {name}")
            label = int(row["label"])
            group = groups.setdefault(name, {"label": label, "boxes": []})
            assert label in (0, 1) and group["label"] == label
            if label:
                x, y, w, h = [int(row[k]) for k in ("x", "y", "width", "height")]
                assert min(x, y) >= 0 and min(w, h) > 0
                group["boxes"].append([x, y, x + w, y + h])
        assert len(groups) == 4882
        assert sum(g["label"] for g in groups.values()) == 1134
        assert sum(len(g["boxes"]) for g in groups.values()) == 1476
        # Require the full public collection before constructing a positive-only
        # grounding split. This is not a classification evaluation subset.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            good = list(pool.map(lambda n: (processed / "images" / n).is_file(), groups))
        if not all(good):
            catalog["unavailable"]["node21"] = {"status": "source_copy_incomplete", "source_expected_images": 4882,
                                                      "source_present_images": sum(good)}
            print(f"PENDING node21: {sum(good)}/4882 public source images", flush=True)
        else:
            import cv2
            convert = load_mha_converter()
            converter = ROOT / "cxrclip/data/dataformat_converter.py"
            converter_sha = sha(converter)
            output_images = ROOT / "local_assets/data_prepared/author_default512/Node21"
            output_images.mkdir(parents=True, exist_ok=True)
            names = sorted(n for n, g in groups.items() if g["label"])

            def process(name):
                original = processed / "images" / name
                target = output_images / (Path(name).stem + ".png")
                receipt = target.with_suffix(".json")
                h, w = header_hw(original)
                before = sha(original)
                if target.exists() or receipt.exists():
                    record = json.loads(receipt.read_text())
                    assert record["source_sha256"] == before and record["png_sha256"] == sha(target)
                    newh, neww = record["png_shape"]
                else:
                    temporary = target.with_name(target.stem + ".partial.png")
                    convert(str(original), str(temporary), do_resize=True)
                    pixels = cv2.imread(str(temporary), cv2.IMREAD_GRAYSCALE)
                    if pixels is None or pixels.ndim != 2 or abs(min(pixels.shape) - 512) > 1:
                        raise ValueError(f"Invalid MHA conversion {name}")
                    newh, neww = pixels.shape
                    temporary.replace(target)
                    record = {"source": str(original), "source_sha256": before, "png": str(target),
                              "png_sha256": sha(target), "source_shape": [h, w], "png_shape": [newh, neww],
                              "converter_sha256": converter_sha, "do_resize": True, "interpolation": "INTER_LANCZOS4"}
                    atomic_json(receipt, record)
                boxes = []
                for x1, y1, x2, y2 in groups[name]["boxes"]:
                    assert x2 <= w and y2 <= h
                    boxes.append([x1 * neww / w, y1 * newh / h, x2 * neww / w, y2 * newh / h])
                return {"image": str(target), "class": repr(["Nodule"] * len(boxes)), "label": "[1]", "boxes": repr(boxes)}, record

            rows, receipts = [], []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, (row, receipt) in enumerate(pool.map(process, names), 1):
                    rows.append(row)
                    receipts.append(receipt)
                    if i % 100 == 0:
                        print(f"CONVERT node21: {i}/{len(names)}", flush=True)
            ledger = args.output / "node21_conversion.json"
            atomic_json(ledger, receipts)
            publish("node21", rows, ["Nodule"], "grounding", "Node21 public-source reconstruction", 1134,
                    "1134 positive images / 1476 real boxes from public source; AlphaRAD's test CSV is unpublished",
                    [source, converter, ledger],
                    ["Full public source coverage checked: 4882 images; only positive images enter pointing-game metric.",
                     "This is not the hidden challenge test set or a recovered AlphaRAD test split.",
                     "simulated_metadata.csv is never used.",
                     "Author MHA converter defaults retained (short edge512, Lanczos4, per-image uint8 min-max).",
                     "Source xywh boxes converted to xyxy, then scaled by actual output width/height ratios; float precision retained.",
                     "Nodule is a reconstructed class name; author Node21 constant not provided."])

    catalog["status"] = "complete"
    catalog["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(catalog_path, catalog)
    print(f"COMPLETE {catalog_path}", flush=True)


if __name__ == "__main__":
    main()
