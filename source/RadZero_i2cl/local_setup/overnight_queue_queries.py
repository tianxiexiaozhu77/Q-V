#!/usr/bin/env python3
"""Append ready-cache query experiments to the fixed overnight GPU queue."""
import fcntl
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT / "runs/overnight_20261020_2150"
RAW = PROJECT / "local_assets/outputs/overnight_20261020_2150"


def main():
    expected = json.loads((CAMPAIGN / "query_source_snapshot/source_sha256.json").read_text())
    for name, digest in expected.items():
        actual = hashlib.sha256((PROJECT / "local_setup" / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Frozen query dependency changed: {name}")
    queue_path = CAMPAIGN / "queue.json"
    added = []
    with (CAMPAIGN / "queue_edit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads(queue_path.read_text())
        for dataset in ("OpenI", "ChestXray14", "ChestXDet10", "SIIM", "RSNA"):
            cache = RAW / "features" / dataset
            manifest_path = cache / "manifest.json"
            if not manifest_path.exists():
                continue
            if json.loads(manifest_path.read_text()).get("status") != "complete":
                continue
            identifier = "query_refinement_" + dataset
            if any(job["id"] == identifier for job in queue["jobs"]):
                continue
            output = RAW / "query_refinement" / dataset
            queue["jobs"].append({
                "id": identifier, "dataset": dataset,
                "argv": ["bash", "local_setup/python_local.sh",
                         "local_setup/overnight_query_refinement.py",
                         "--cache", str(cache), "--output", str(output),
                         "--batch-size", "16", "--workers", "2", "--device", "cuda",
                         "--etas", "0", ".01", ".03", ".1", ".3",
                         "--deadline", "2026-10-21T10:00:00+08:00"],
                "command_marker": "local_setup/overnight_query_refinement.py",
                "output": str(output), "kind": "query_refinement",
                "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            })
            added.append(dataset)
        if added:
            temporary = queue_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(queue, indent=2) + "\n")
            temporary.replace(queue_path)
    print(json.dumps({"queued_queries": added}))


if __name__ == "__main__":
    main()
