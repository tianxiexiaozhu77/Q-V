#!/usr/bin/env python3
"""Reuse the frozen CheXchoNet patient holdout and supports for eight baselines.

No selection is performed here. Every task keeps the exact ordered image/label
pairs already evaluated by AlphaRAD and RadZero, including its 5+5 supports.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDICAL = ROOT.parent
TASKS = ("slvh", "dlv", "composite_slvh_dlv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def convert(rows, classes):
    converted = []
    for index, row in enumerate(rows):
        labels = ast.literal_eval(row["label"])
        if len(labels) != len(classes) or any(v not in (0, 1) for v in labels):
            raise ValueError("Nonbinary or inconsistent labels")
        converted.append({"index": index, "image": row["image"],
            "labels": labels, "class_names_present": ast.literal_eval(row["class"]),
            "patient_id": row["patient_id"]})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=MEDICAL / "AlphaRAD/local_setup/chexchonet_20261023")
    parser.add_argument("--reference", type=Path, default=MEDICAL / "experiments/chexchonet_comparison_20261023_233926")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/chexchonet_20261023")
    args = parser.parse_args()
    source, reference, output = (p.resolve() for p in (args.source, args.reference, args.output))
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifests: {output}")
    original_catalog = json.loads((source / "catalog.json").read_text())
    split_audit = json.loads((source / "split_audit.json").read_text())
    integrity = json.loads((source / "image_integrity_audit.json").read_text())
    prompts = json.loads((source / "prompts.json").read_text())
    if integrity["status"] != "passed" or integrity["support_patient_test_duplicate_pixel_groups"]:
        raise ValueError("Source image integrity or duplicate audit did not pass")
    all_rows = read_csv(source / "full_evaluation.csv")
    common_rows = read_csv(source / "evaluation.csv")
    paths = [r["image"] for r in common_rows]
    patients = [r["patient_id"] for r in common_rows]
    labels = np.asarray([[int(r[t]) for t in TASKS] for r in common_rows])
    if len(paths) != 7407 or len(set(paths)) != 7407:
        raise ValueError("Expected exactly 7,407 unique matched evaluation images")
    alpha = reference / "alpharad_paired"
    radzero = reference / "radzero_paired"
    if json.loads((alpha / "image_paths.json").read_text()) != paths:
        raise ValueError("AlphaRAD saved image order differs")
    if json.loads((alpha / "patient_ids.json").read_text()) != patients:
        raise ValueError("AlphaRAD saved patient order differs")
    if not np.array_equal(np.load(alpha / "labels.npy", allow_pickle=False), labels):
        raise ValueError("AlphaRAD saved labels differ")
    radzero_rows = read_csv(radzero / "predictions_paired.csv")
    if [r["image_path"] for r in radzero_rows] != paths or [r["patient_id"] for r in radzero_rows] != patients:
        raise ValueError("RadZero saved image/patient order differs")
    if not np.array_equal(np.asarray([[int(r["label:" + t]) for t in TASKS] for r in radzero_rows]), labels):
        raise ValueError("RadZero saved labels differ")
    inventory = {r["image"]: r for r in integrity["inventory"]}
    if set(inventory) != {r["image"] for r in all_rows}:
        raise ValueError("Source integrity inventory does not cover exactly the full subset")
    print("Validating SHA256 of all 7,494 frozen source images...", flush=True)
    def verify(path):
        digest = sha(path)
        if digest != inventory[path]["sha256"]:
            raise ValueError(f"Source image changed: {path}")
        return path, digest
    with ThreadPoolExecutor(max_workers=8) as pool:
        image_hashes = dict(pool.map(verify, inventory))
    support_union = set()
    for task in TASKS:
        support_union.update(r["patient_id"] for r in read_csv(source / task / "support.csv"))
    if support_union & set(patients) or len(support_union) != 30:
        raise ValueError("Support/test patient separation changed")
    if [r["image"] for r in all_rows if r["patient_id"] not in support_union] != paths:
        raise ValueError("Common evaluation is not the exact patient-excluded full subset")
    sources = [source / name for name in ("catalog.json", "split_audit.json", "image_integrity_audit.json", "prompts.json", "evaluation.csv", "full_evaluation.csv")]
    sources += [alpha / name for name in ("image_paths.json", "patient_ids.json", "labels.npy", "result.json")]
    sources += [radzero / "predictions_paired.csv", radzero / "result.json", Path(__file__).resolve()]
    sources_sha = {str(p): sha(p) for p in sources}
    output.mkdir(parents=True)
    write(output / "prompts.json", prompts)
    write(output / "image_sha256.json", image_hashes)
    catalog = {"status": "complete", "version": "chexchonet-fixed-support-v1", "eta": .3,
        "views": ["identity", "center80"], "negative_query_policy": "unchanged",
        "query_norm_policy": "preserve original norm after direction update",
        "scope": "CheXchoNet custom subset; 3 binary tasks; exact AlphaRAD/RadZero patient holdout",
        "source_catalogs": {str(source / "catalog.json"): sha(source / "catalog.json")},
        "dataset_count_unique": 1, "task_count": 3, "classification_task_count": 3,
        "prompt_source": "Fixed semantic disease names; native baseline prompt templates preserved. Source author CheXchoNet prompts unavailable.",
        "fixed_semantic_prompts": prompts["tasks"],
        "baseline_reference_policy": "Original branch computed with native frozen adapter in this evaluation; no prior eight-baseline results exist for this dataset.",
        "datasets": {}}
    full_catalog = {"status": "complete", "scope": "Full 7494 custom subset; pretrained original-only optional evaluation; never paired with reduced-test adapted metrics", "datasets": {}}
    notes = list(split_audit["limitations"]) + [
        "This transfer preserves each baseline's native preprocessing and prompt template; names are explicit cardiac task semantics, not cardiomegaly proxies.",
        "The public images are already CLAHE-processed 224x224 images; resizing cannot recover original resolution.",
        "Every baseline and branch uses the exact same 7407-image patient holdout already evaluated by AlphaRAD/RadZero."]
    for index, task in enumerate(TASKS):
        key = "chexchonet_" + task
        entry = original_catalog["datasets"][key]
        for field, digest_key in (("csv_path", "csv_sha256"), ("support_csv_path", "support_csv_sha256"), ("original_csv_path", "original_csv_sha256")):
            if sha(entry[field]) != entry[digest_key]:
                raise ValueError("Frozen source CSV changed: " + field)
        classes = entry["class_names"]
        rows = convert(read_csv(entry["csv_path"]), classes)
        supports = convert(read_csv(entry["support_csv_path"]), classes)
        full = convert(read_csv(entry["original_csv_path"]), classes)
        if [r["image"] for r in rows] != paths or [r["labels"][0] for r in rows] != labels[:, index].tolist():
            raise ValueError("Task-specific image/label order mismatch")
        if len(supports) != 10 or sum(r["labels"][0] for r in supports) != 5:
            raise ValueError("Expected fixed five positive + five negative supports")
        task_sources = {str(Path(entry[f])): sha(entry[f]) for f in ("csv_path", "support_csv_path", "original_csv_path", "split_audit_path")}
        task_sources.update({str(source / "prompts.json"): sha(source / "prompts.json")})
        sources_sha.update(task_sources)
        refs = [{"model": model, "result": str(directory / "result.json"), "sha256": sha(directory / "result.json"),
                 "candidate": "tta_center_refined", "source_label_column": task}
                for model, directory in (("AlphaRAD", alpha), ("RadZero", radzero))]
        manifest = {"dataset": key, "task": "classification", "class_names": classes,
            "rows": rows, "count": len(rows), "source_label_column": task,
            "source": {"csv": entry["csv_path"], "sha256": entry["csv_sha256"]},
            "comparison_references": refs, "notes": notes,
            "sampling": "No resampling: exact frozen support-patient-excluded test order and labels.",
            "fixed_semantic_prompt": prompts["tasks"][task]}
        dest = output / (key + ".json")
        write(dest, manifest)
        for row in supports:
            row["byte_sha256"] = image_hashes[row["image"]]
        support_manifest = {**{k: v for k, v in manifest.items() if k != "rows"},
            "support_rows": supports, "support_count": 10, "support_positive_counts": [5],
            "support_negative_counts": [5], "test_manifest": str(dest), "test_manifest_sha256": sha(dest),
            "support_sources": task_sources, "baseline_references": {},
            "baseline_reference_policy": "Compute native baseline in the paired worker; no saved eight-baseline results available yet.",
            "identity_audit": {"patient_id": {"available_on_both": True, "overlap_count": 0}},
            "support_policy": "Exact previous 5-positive/5-negative images per task; no resampling; original-view support features only."}
        support_dest = output / (key + "_support.json")
        write(support_dest, support_manifest)
        catalog["datasets"][key] = {"dataset": key, "task": "classification", "manifest": str(dest),
            "sha256": sha(dest), "count": len(rows), "class_names": classes, "comparison_references": refs,
            "support_manifest": str(support_dest), "support_manifest_sha256": sha(support_dest), "support_count": 10}
        full_dest = output / (key + "_full.json")
        write(full_dest, {**manifest, "rows": full, "count": len(full), "source": {"csv": entry["original_csv_path"],
            "sha256": entry["original_csv_sha256"]}, "comparison_references": [],
            "sampling": "Full original custom subset; not a matched comparison with adapted branch."})
        full_catalog["datasets"][key] = {"dataset": key, "task": "classification", "manifest": str(full_dest),
            "sha256": sha(full_dest), "count": len(full), "class_names": classes, "comparison_references": []}
    audit = {"status": "passed", "full_image_count": 7494, "test_image_count": 7407,
        "test_patient_count": len(set(patients)), "excluded_patient_count": len(support_union),
        "excluded_image_count": len(all_rows) - len(paths), "support_per_task": {"positive": 5, "negative": 5},
        "patient_overlap_count": 0, "full_image_sha256_rechecked": len(image_hashes),
        "exact_AlphaRAD_saved_image_order_labels_patients": True,
        "exact_RadZero_saved_image_order_labels_patients": True,
        "ordered_image_paths_sha256": hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest(),
        "ordered_labels_sha256": hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest(),
        "test_positive_counts": dict(zip(TASKS, labels.sum(axis=0).tolist())), "source_sha256": sources_sha}
    write(output / "audit.json", audit)
    write(output / "full_catalog.json", full_catalog)
    write(output / "catalog.json", catalog)
    (output / "DATASETS.txt").write_text(
        "CheXchoNet 自定义 7,494 张；不是论文原始划分。\n"
        "复用 AlphaRAD/RadZero 原有支持集，不重抽；每任务 5 正＋5 负。\n"
        "排除三任务合计 30 个支持患者的全部 87 张图像，统一测试 7,407 张（2,470 患者）。\n"
        "CSLV: 567 正 / 6,840 负；CDLV: 315 正 / 7,092 负；CCOM: 828 正 / 6,579 负。\n"
        "已核对两模型既有输出：图像顺序、患者、标签完全一致；重新验证全部 7,494 张 SHA256。\n"
        "catalog.json 用于同集原始/改进公平对照；full_catalog.json 仅供可选原始完整集评估。\n"
        "baseline 使用各自原有文本模板和评分方式；任务语义使用明确心脏疾病名称，不替换成心脏增大。\n")
    print(json.dumps({"catalog": str(output / "catalog.json"), "audit": str(output / "audit.json"),
        "tasks": 3, "test_images_per_task": 7407, "support_images_per_task": 10}, indent=2), flush=True)


if __name__ == "__main__":
    main()
