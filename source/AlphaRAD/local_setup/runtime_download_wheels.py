#!/usr/bin/env python3
"""Download pinned Linux wheels directly from PyPI with 16 resumable ranges."""

import concurrent.futures
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import urllib.request


ROOT = Path("/nvme1/zjx/code/medical/alpharad-assets/downloads/model_runtime/wheels")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHUNK_SIZE = 8 * 1024 * 1024
import re
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
PROJECTS = [tuple(line.strip().split("==")) for line in Path("/nvme1/zjx/code/medical/AlphaRAD/local_setup/runtime-resolved.txt").read_text().splitlines() if re.fullmatch(r"[a-zA-Z0-9_.-]+==[^ ]+", line)]
SUPPORTED_TAGS = {tag:i for i, tag in enumerate(sys_tags())}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(name, version):
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    with OPENER.open(metadata_url, timeout=60) as response:
        info = json.load(response)
    candidates = []
    for entry in info["urls"]:
        if entry["filename"].endswith(".whl"):
            tags = parse_wheel_filename(entry["filename"])[3]
            priority = min((SUPPORTED_TAGS[tag] for tag in tags if tag in SUPPORTED_TAGS), default=None)
            if priority is not None:
                candidates.append((priority, entry))
    if candidates:
        entry = min(candidates, key=lambda x:x[0])[1]
    else:
        entry = next(e for e in info["urls"] if e["packagetype"] == "sdist")
    return {
        "project": name, "version": version, "filename": entry["filename"],
        "url": entry["url"], "size": entry["size"],
        "sha256": entry["digests"]["sha256"], "metadata_url": metadata_url,
    }


def fetch_chunk(entry, index):
    begin = index * CHUNK_SIZE
    end = min(begin + CHUNK_SIZE, entry["size"]) - 1
    length = end - begin + 1
    parts = ROOT / (entry["filename"] + ".parts")
    parts.mkdir(exist_ok=True)
    target = parts / f"{index:06d}"
    temporary = target.with_suffix(".part")
    if target.exists() and target.stat().st_size == length:
        return entry["filename"], length
    error = None
    for attempt in range(8):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset > length:
                raise RuntimeError(f"Oversized partial chunk {temporary}")
            if offset == length:
                temporary.replace(target)
                return entry["filename"], length
            request = urllib.request.Request(entry["url"], headers={
                "Range": f"bytes={begin + offset}-{end}",
                "User-Agent": "AlphaRAD-local-setup/1.0",
            })
            with OPENER.open(request, timeout=60) as response:
                expected = f"bytes {begin + offset}-{end}/{entry['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Unexpected range response: {response.status} {response.headers.get('Content-Range')}")
                with temporary.open("ab") as handle:
                    while True:
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if temporary.stat().st_size != length:
                raise RuntimeError(f"Incomplete chunk: {temporary.stat().st_size}/{length}")
            temporary.replace(target)
            return entry["filename"], length
        except Exception as exc:
            error = exc
            if attempt < 7:
                time.sleep(min(attempt + 1, 8))
    raise RuntimeError(f"{entry['filename']} chunk {index} failed: {error}")


def finalize(entry):
    target = ROOT / entry["filename"]
    temporary = target.with_suffix(target.suffix + ".assembling")
    parts = ROOT / (entry["filename"] + ".parts")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(math.ceil(entry["size"] / CHUNK_SIZE)):
            with (parts / f"{index:06d}").open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
    if temporary.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise RuntimeError(f"PyPI SHA256 verification failed for {target}")
    temporary.replace(target)
    for index in range(math.ceil(entry["size"] / CHUNK_SIZE)):
        (parts / f"{index:06d}").unlink()
    parts.rmdir()
    entry["verified"] = True
    print(json.dumps({"completed": str(target), "bytes": entry["size"], "sha256": entry["sha256"], "verified": True}), flush=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(lambda item: metadata(*item), PROJECTS))
    manifest = ROOT / "pypi-wheel-manifest.json"
    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    for entry in entries:
        print(json.dumps(entry), flush=True)
    pending = []
    for entry in entries:
        wheel = ROOT / entry["filename"]
        if wheel.is_file() and wheel.stat().st_size == entry["size"] and sha256_file(wheel) == entry["sha256"]:
            entry["verified"] = True
            print(f"Already SHA256 verified: {wheel.name}", flush=True)
        else:
            pending.append(entry)
    lookup = {entry["filename"]: entry for entry in pending}
    totals = {entry["filename"]: math.ceil(entry["size"] / CHUNK_SIZE) for entry in pending}
    completed_chunks = {key: 0 for key in totals}
    completed_bytes = {key: 0 for key in totals}
    task_lists = [[(entry, index) for index in range(totals[entry["filename"]])] for entry in pending]
    interleaved = [task for batch in itertools.zip_longest(*task_lists) for task in batch if task is not None]
    started = time.monotonic()
    last_report = started
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch_chunk, entry, index) for entry, index in interleaved]
        for future in concurrent.futures.as_completed(futures):
            filename, downloaded = future.result()
            completed_bytes[filename] += downloaded
            completed_chunks[filename] += 1
            now = time.monotonic()
            if now - last_report >= 10:
                mib = sum(completed_bytes.values()) / 2**20
                print(json.dumps({"downloaded_MiB": round(mib, 1), "elapsed_seconds": round(now - started, 1), "MiB_per_second": round(mib / (now - started), 2), "completed_chunks": completed_chunks}), flush=True)
                last_report = now
            if completed_chunks[filename] == totals[filename]:
                finalize(lookup[filename])
                manifest.write_text(json.dumps(entries, indent=2) + "\n")
    manifest.write_text(json.dumps(entries, indent=2) + "\n")
    print("All wheels have been verified against official PyPI SHA256 hashes.", flush=True)


if __name__ == "__main__":
    main()
