"""Index all paper benchmarks without conflating source data and author splits."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "alpharad-assets"
IMAGES = ASSETS / "data_images"
SMALL = ASSETS / "downloads/public_small"
PAPER = "https://arxiv.org/html/2609.01757v1"

# name, tasks, paper test size, local source family, access / preparation state
ROWS = [
    ("ChestXDet10", ["classification", "grounding"], 542, "ChestXDet10", "reused_images"),
    ("CheXpert", ["classification"], 668, "CheXpert", "requires_access"),
    ("ChestXray14", ["classification"], 22433, "ChestXray14", "reused_images"),
    ("CheXpert5x200", ["classification"], 1000, "CheXpert5x200", "metadata_only"),
    ("VinDr-CXR", ["classification", "grounding"], 3000, "VinDr-CXR", "requires_access"),
    ("CXR-LT Task 2", ["classification"], 406, "CXR-LT", "requires_access"),
    ("CXR-LT Task 3", ["classification"], 78946, "CXR-LT", "requires_access"),
    ("PadChest", ["classification"], 39053, "PadChestSample", "public_sample_only"),
    ("CheXchoNet SLVH", ["classification"], 7494, "CheXchoNet", "requires_access"),
    ("CheXchoNet DLV", ["classification"], 7494, "CheXchoNet", "requires_access"),
    ("CheXchoNet Composite", ["classification"], 7494, "CheXchoNet", "requires_access"),
    ("RSNA Pneumonia", ["classification"], 5337, "RSNA", "reused_images"),
    ("SIIM-ACR", ["classification", "segmentation"], 1704, "SIIM", "reused_images"),
    ("ShenzhenXray", ["classification"], 663, "ShenzhenXray", "public_source"),
    ("Montgomery", ["classification"], 139, "Montgomery", "public_source"),
    ("Covid-Kaggle", ["classification"], 1382, "CovidKaggle", "public_source"),
    ("CheXlocalize", ["grounding", "segmentation"], 1535, "CheXlocalizeSample", "public_sample_only"),
    ("Node21", ["grounding"], 1476, "Node21", "public_source"),
    ("MS-CXR V1", ["phrase_grounding"], 167, "MS-CXR", "requires_access"),
    ("MS-CXR V2", ["phrase_grounding"], 216, "MS-CXR", "requires_access"),
    ("PadChest-GR", ["phrase_grounding"], 4555, "PadChest-GR", "requires_access"),
    ("QaTa-COV19", ["segmentation"], 2113, "QaTaCOV19", "public_source"),
]


def main():
    csvs = json.loads((ROOT / "local_setup/data_status.json").read_text())["author_csvs"]
    author = {"ChestXDet10": "chestxdet10_gt", "CheXpert": "chexpert",
              "Covid-Kaggle": "covidkaggle", "CheXlocalize": "chexlocalize_seg",
              "QaTa-COV19": "qatacovid_seg"}
    entries = []
    for name, tasks, size, family, state in ROWS:
        entry = {"name": name, "tasks": tasks, "paper_test_size": size,
                 "local_family": family, "status": state,
                 "local_source_path": str(IMAGES / family),
                 "alpharad_author_csv_released": name in author}
        if name in author:
            entry["author_csv_coverage"] = csvs[author[name]]
        if family in ("ShenzhenXray", "Montgomery", "Node21"):
            manifest = SMALL / f"{family}_manifest.json"
            if manifest.is_file():
                source = json.loads(manifest.read_text())
                entry.update(status="public_source_downloaded", source_image_count=source["images"],
                             source_manifest=str(manifest))
            else:
                entry["status"] = "public_source_download_in_progress"
        if name == "Node21":
            entry["note"] = "4882-image public training source: 1476 positive boxes across 1134 positive images, plus 3748 negative images. The box count matches the paper's 1476 cases, but AlphaRAD's evaluation CSV is unpublished, so exact selection and row order cannot be verified."
        elif name == "PadChest":
            entry["note"] = "24-image official-referred sample; full evaluation images still require access."
        elif name == "CheXlocalize":
            entry["note"] = "Two official validation sample JPGs; these do not replace the 499 unique images in the released AlphaRAD segmentation CSV."
        elif name in ("RSNA Pneumonia", "SIIM-ACR", "ChestXray14"):
            entry["note"] = "Original images and RadZero reference annotations reused. AlphaRAD does not ship this benchmark's evaluation CSV/config in its sample release."
        entries.append(entry)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "paper": PAPER,
              "training": {"dataset": "MIMIC-CXR", "status": "requires_access",
                           "source": "https://physionet.org/content/mimic-cxr-jpg/2.1.0/",
                           "image_files_downloaded": False,
                           "alpharad_200k_concepts_downloaded": False,
                           "complete_training_implementation_released": False},
              "test_benchmark_count": len(entries), "test_benchmarks": entries,
              "notes": ["Benchmark names include task/version variants of the same source dataset.",
                        "Source images, masks, CSV rows and independent image counts are distinct quantities.",
                        "See SETUP_LOCAL.md for current official access links and reproduction limits."]}
    output = ASSETS / "manifests/dataset_catalog.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Indexed training data and {len(entries)} test benchmarks: {output}")


if __name__ == "__main__":
    main()
