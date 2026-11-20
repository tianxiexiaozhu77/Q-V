"""Publish complete COVID/Montgomery manifests as external copies arrive."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data/alpharad"
LOCAL = ROOT / "local_setup"
STATE = LOCAL / "remaining_data_watch.json"
STOP = LOCAL / "remaining_data_watch.STOP"


def save(state):
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)


def main():
    source = ROOT / "configs/data_csv/covidkaggle.csv"
    with source.open(newline="") as handle:
        covid_paths = [DATA / "CovidKaggle" / r["image"].split("/covid_kaggle_256/", 1)[1]
                       for r in csv.DictReader(handle)]
    state = {"pid": os.getpid(), "status": "waiting", "datasets": {}, "completed": []}
    save(state)
    while True:
        if STOP.exists():
            state["status"] = "stopped_by_marker"
            save(state)
            return
        unified = LOCAL / "pretrained_catalog.json"
        if unified.exists():
            catalog = json.loads(unified.read_text())
            if catalog.get("status") == "complete":
                state["status"] = "stopped_catalog_finalized"
                save(state)
                return
        else:
            catalog = {"datasets": {}}
        for key in ("covidkaggle", "montgomery"):
            if key in catalog["datasets"] or key in state["completed"]:
                if key not in state["completed"]:
                    state["completed"].append(key)
                continue
            if key == "covidkaggle":
                with ThreadPoolExecutor(max_workers=4) as pool:
                    present = sum(pool.map(Path.is_file, covid_paths))
                expected = len(covid_paths)
            else:
                present = len(list((DATA / "Montgomery").rglob("CXR_png/MCUCXR_*.png")))
                expected = 138
            state["datasets"][key] = {"expected": expected, "present": present,
                                        "status": "complete_source" if present == expected else "copy_in_progress"}
            save(state)
            if present != expected:
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = LOCAL / f"data_20261021_additional_auto_{key}_{stamp}"
            print(f"PREPARING {key}: {present}/{expected}", flush=True)
            subprocess.run([sys.executable, str(LOCAL / "prepare_storage_additional.py"),
                            "--output", str(output), "--datasets", key], check=True)
            # If the operator finalized during preparation, leave the immutable
            # new list available but do not reopen the evaluation catalog.
            if unified.exists() and json.loads(unified.read_text()).get("status") == "complete":
                state["status"] = "stopped_catalog_finalized"
                save(state)
                return
            subprocess.run([sys.executable, str(LOCAL / "merge_prepared_catalogs.py")], check=True)
            state["completed"].append(key)
            state["datasets"][key].update(status="published", catalog=str(output / "catalog.json"))
            save(state)
        if len(state["completed"]) == 2:
            state["status"] = "complete"
            save(state)
            print("COMPLETE remaining two datasets published", flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(status="failed", error=repr(exc))
        save(state)
        raise
