#!/usr/bin/env python3
"""Collect completed manifests; image files and source CSVs remain untouched."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finalize", action="store_true", help="No more manifests are expected in this run")
    args = p.parse_args()
    paths = [ROOT / "local_setup/data_20261021_complete/catalog.json"]
    paths += sorted((ROOT / "local_setup").glob("data_20261021_additional*/catalog.json"), key=lambda p: p.stat().st_mtime)
    merged = {"status": "complete" if args.finalize else "preparing", "datasets": {}, "unavailable": {},
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_catalogs": []}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        merged["source_catalogs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for key, entry in data.get("datasets", {}).items():
            csv = Path(entry["csv_path"])
            if hashlib.sha256(csv.read_bytes()).hexdigest() != entry["csv_sha256"]:
                raise ValueError(f"CSV hash mismatch: {csv}")
            merged["datasets"][key] = entry
        merged["unavailable"].update(data.get("unavailable", {}))
    available_names = {v["display_name"].lower() for v in merged["datasets"].values()}
    available_names.update(key.lower() for key in merged["datasets"])
    for key in list(merged["unavailable"]):
        if key in merged["datasets"] or key.lower() in available_names:
            del merged["unavailable"][key]
    output = ROOT / "local_setup/pretrained_catalog.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
    temp.replace(output)
    print(json.dumps({"catalog": str(output), "datasets": list(merged["datasets"]), "status": merged["status"]}))


if __name__ == "__main__":
    main()
