#!/usr/bin/env python3
"""Freeze existing phase6 holdouts for pretrained baselines, without resampling."""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDICAL = ROOT.parent
ALPHA = MEDICAL / "AlphaRAD"
CAMPAIGN = MEDICAL / "experiments/query_search_20261021_152010"
RZ_SPLITS = MEDICAL / "RadZero_i2cl/local_assets/splits/native_public5_local_seed42"
CHEXPERT = ALPHA / "runs/chexpert_phase6_20261022_170123_2309385"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def parsed(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def publish(output, key, rows, classes, task, source, refs, notes, identity=None):
    paths = [r["image"] for r in rows]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"Empty/duplicate image paths: {key}")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{key}: {len(missing)} missing images, e.g. {missing[:2]}")
    record = []
    for i, row in enumerate(rows):
        item = {"index": i, "image": row["image"], "class_names_present": parsed(row["class"]) if row["class"].startswith("[") else [row["class"]]}
        if task == "classification":
            y = parsed(row["label"])
            y = y if isinstance(y, list) else [y]
            if len(y) != len(classes) or any(v not in (0, 1) for v in y):
                raise ValueError(f"Invalid target: {key}/{i}")
            item["labels"] = [int(v) for v in y]
        elif task == "grounding":
            item["boxes"] = parsed(row["boxes"])
            if len(item["boxes"]) != len(item["class_names_present"]):
                raise ValueError(f"Box/label count differs: {key}/{i}")
        elif task == "segmentation":
            item["encoded_pixels"] = parsed(row["EncodedPixels"])
        else:
            raise ValueError(task)
        if identity:
            item.update(identity[i])
        else:
            for field in ("patient_id", "study_id", "group"):
                if row.get(field):
                    item[field] = row[field]
        record.append(item)
    manifest = {"dataset": key, "task": task, "class_names": classes, "rows": record,
        "count": len(rows), "source": source, "comparison_references": refs, "notes": notes,
        "sampling": "unchanged previously frozen evaluation rows; baseline receives no support examples",
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    path = output / f"{key}.json"
    atomic_json(path, manifest)
    return {"dataset": key, "task": task, "manifest": str(path), "sha256": sha(path),
        "count": len(rows), "class_names": classes, "comparison_references": refs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/matched_phase6_20261022")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog_path = ALPHA / "runs/query_refinement_20261021_1441/catalog.json"
    entries = json.loads(catalog_path.read_text())["datasets"]
    chexpert_path = CHEXPERT / "data/catalog.json"
    entries.update(json.loads(chexpert_path.read_text())["datasets"])
    frozen = {}
    shared_rz = {"chestxdet10": "ChestXDet10", "chest14": "ChestXray14", "rsna_pneumonia": "RSNA"}
    for key, entry in entries.items():
        csv_path = Path(entry["csv_path"])
        if sha(csv_path) != entry["csv_sha256"]:
            raise ValueError(f"Changed old split CSV: {csv_path}")
        rows = read_csv(csv_path)
        if len(rows) != entry["actual_count"]:
            raise ValueError(f"Changed old split count: {key}")
        ref = CHEXPERT / "paired/result.json" if key == "chexpert" else CAMPAIGN / f"runs/phase6/alpharad_{key}/result.json"
        result = json.loads(ref.read_text())
        if result.get("status") != "complete" or result["rows"] != len(rows):
            raise ValueError(f"Incomplete/mismatched reference: {ref}")
        refs = [{"model": "AlphaRAD", "result": str(ref), "sha256": sha(ref), "candidate": "tta_center_refined"}]
        if key in shared_rz:
            rz_name = shared_rz[key]
            rz_path = RZ_SPLITS / rz_name / "test.csv"
            rz_rows = read_csv(rz_path)
            # Same ordered image IDs and labels across the independently stored manifests.
            identity_key = "patient_id" if rz_name == "RSNA" else "image_id"
            if ([Path(r["image"]).stem for r in rows] != [r[identity_key] for r in rz_rows]
                    or [parsed(r["label"]) for r in rows] != [json.loads(r["labels_json"]) for r in rz_rows]):
                raise ValueError(f"Cross-model holdout identities/labels differ: {key}")
            rz_ref = CAMPAIGN / f"runs/phase6/radzero_{rz_name}/result.json"
            refs.append({"model": "RadZero", "result": str(rz_ref), "sha256": sha(rz_ref),
                "candidate": "tta_center_refined", "matched_rz_csv": str(rz_path), "matched_rz_csv_sha256": sha(rz_path)})
        frozen[key] = publish(output, key, rows, entry["class_names"], entry["task"],
            {"csv": str(csv_path), "sha256": sha(csv_path), "support_count_excluded_from_baseline_input": entry["support_count"]},
            refs, entry.get("notes", []))
    # OpenI is evaluated only by the RadZero arm; preserve exact report-held-out rows.
    key = "openi"
    split = RZ_SPLITS / "OpenI"
    m = json.loads((split / "manifest.json").read_text())
    rz_rows = read_csv(split / "test.csv")
    classes = m["class_names"]
    rows = []
    for row in rz_rows:
        y = json.loads(row["labels_json"])
        rows.append({"image": str(MEDICAL / "data/radzero/prepared" / row["image_path"]),
            "label": repr(y), "class": repr([c for c, v in zip(classes, y) if v]),
            **{k: row.get(k, "") for k in ("patient_id", "study_id", "group")}})
    ref = CAMPAIGN / "runs/phase6/radzero_OpenI/result.json"
    assert len(rows) == json.loads(ref.read_text())["rows"] == 7294
    frozen[key] = publish(output, key, rows, classes, "classification",
        {"csv": str(split / "test.csv"), "sha256": sha(split / "test.csv"), "support_count_excluded_from_baseline_input": 90},
        [{"model": "RadZero", "result": str(ref), "sha256": sha(ref), "candidate": "tta_center_refined"}], m["limitations"])
    # SIIM uses different support sources across the two reference models.
    # Use the exact RadZero subset, mapped by DICOM UID to the same audited PNG conversion as AlphaRAD.
    alpha_siim = read_csv(entries["siim_pneumothorax"]["csv_path"])
    by_uid = {Path(row["image"]).stem: row for row in alpha_siim}
    siim_csv = RZ_SPLITS / "SIIM/test.csv"
    rz_rows = read_csv(siim_csv)
    rows, identity = [], []
    for r in rz_rows:
        row = by_uid[r["image_id"]]
        if parsed(row["label"]) != json.loads(r["labels_json"]):
            raise ValueError("SIIM mapped labels differ")
        rows.append(row)
        identity.append({k: r[k] for k in ("patient_id", "study_id", "group")})
    ref = CAMPAIGN / "runs/phase6/radzero_SIIM/result.json"
    assert len(rows) == json.loads(ref.read_text())["rows"] == 1694
    frozen["siim_radzero_holdout"] = publish(output, "siim_radzero_holdout", rows, ["Pneumothorax"], "classification",
        {"csv": str(siim_csv), "sha256": sha(siim_csv), "images": "AlphaRAD author-converted PNG, exact DICOM UID/label match", "support_count_excluded_from_baseline_input": 10},
        [{"model": "RadZero", "result": str(ref), "sha256": sha(ref), "candidate": "tta_center_refined"}],
        ["RadZero patient holdout: 1694 images; distinct from AlphaRAD's 1704-image SIIM test scope."], identity)
    catalog = {"status": "complete", "datasets": frozen,
        "dataset_count_unique": 11, "task_count": len(frozen), "classification_task_count": sum(e["task"] == "classification" for e in frozen.values()),
        "baseline_support_usage": "none; pretrained baselines use test images only",
        "scope": "AlphaRAD 10 datasets/11 tasks plus OpenI and RadZero SIIM subset",
        "source_script_sha256": sha(__file__)}
    atomic_json(output / "catalog.json", catalog)
    lines = ["固定测试范围：不重新抽样，不将支持样本交给零样本baseline。", "",
        "| 任务 | 类型 | 图片数 | 类别数 |", "|---|---|---:|---:|"]
    lines += [f"| {k} | {v['task']} | {v['count']} | {len(v['class_names'])} |" for k, v in frozen.items()]
    (output / "DATASETS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"catalog": str(output / "catalog.json"), "tasks": len(frozen), "counts": {k: v["count"] for k, v in frozen.items()}}, indent=2))


if __name__ == "__main__":
    main()
