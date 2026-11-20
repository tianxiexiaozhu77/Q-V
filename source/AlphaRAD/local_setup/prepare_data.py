"""Create local copies of author CSVs, changing image paths only.

Does not invent missing evaluation rows or reproduce unpublished preprocessing.
Original CSVs and labels remain untouched under configs/data_csv.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "alpharad-assets"
IMAGES = ASSETS / "data_images"
LOCAL = ROOT / "local_setup"
SPECS = {
    "chestXDet10_gt.csv": ("chestxdet10_gt", 10, "ChestXDet10", "/chestXDet10/", None),
    "covidkaggle.csv": ("covidkaggle", 1, "CovidKaggle", "/covid_kaggle_256/", 1382),
    "qatacov19.csv": ("qatacovid_seg", 1, "QaTaCOV19", "/qatarCovid/", 2113),
    "chexpert.csv": ("chexpert", 12, "CheXpert/test", ".../", 668),
    "chestXlocalize_seg.csv": ("chexlocalize_seg", 10, "CheXlocalize", "/chexlocalize/", 1535),
}


def main():
    output = LOCAL / "data_csv"
    output.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "author_csvs": {}}
    config = {"base": {"seed": 1234}, "data_test": {},
              "dataloader": {"test": {"batch_size": 8, "num_workers": 2,
                                          "pin_memory": False, "shuffle": False,
                                          "drop_last": False, "prefetch_factor": 2}},
              "test": {"seed": 1234, "checkpoint": []}, "seg_save_dir": None}
    for filename, (name, nclass, folder, marker, paper_size) in SPECS.items():
        original = ROOT / "configs/data_csv" / filename
        with original.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows, fields = list(reader), reader.fieldnames
        paths = []
        for row in rows:
            if marker not in row["image"]:
                raise ValueError(f"Unexpected path in {filename}: {row['image']}")
            relative = Path(row["image"].split(marker, 1)[1])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe relative image path: {relative}")
            row["image"] = str(IMAGES / folder / relative)
            paths.append(row["image"])
        local_csv = output / filename
        with local_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        unique = sorted(set(paths))
        missing = [path for path in unique if not Path(path).is_file()]
        entry = {"rows": len(rows), "unique_images": len(unique),
                 "present_images": len(unique) - len(missing),
                 "missing_images": len(missing), "complete": not missing,
                 "original_csv": str(original), "local_csv": str(local_csv),
                 "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                 "local_sha256": hashlib.sha256(local_csv.read_bytes()).hexdigest(),
                 "missing_examples": missing[:3]}
        if paper_size is not None:
            entry["paper_reported_size"] = paper_size
        if name == "covidkaggle":
            entry["preprocessing_note"] = "Original public 299x299 images. Author paths refer to an unreleased covid_kaggle_256 copy; exact paper preprocessing is not established."
        if name == "chexpert":
            entry["label_note"] = "Released CSV and constants contain 12 classes; paper describes 14. Kept author labels unchanged."
        report["author_csvs"][name] = entry
        if not missing:
            config["data_test"][name] = {"name": name, "data_type": "image_classification",
                                           "data_path": str(local_csv), "n_class": nclass,
                                           "normalize": None}
        print(f"{name}: {len(unique)-len(missing)}/{len(unique)} images; {len(rows)} CSV rows")
    checkpoints = sorted((ASSETS / "model").rglob("*.tar"))
    config["test"]["checkpoint"] = [str(path) for path in checkpoints]
    # JSON is also YAML; OmegaConf can load this without additional templating.
    (LOCAL / "eval_local.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (LOCAL / "data_status.json").write_text(json.dumps(report, indent=2) + "\n")
    (ASSETS / "manifests/author_csv_coverage.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
