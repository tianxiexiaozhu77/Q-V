"""Archive primary-source access pages and the AlphaRAD paper (no credentials)."""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "alpharad-assets"
URLS = {
    "papers/AlphaRAD-ECCV2026.pdf": "https://arxiv.org/pdf/2609.01757",
    "sources/paper.html": "https://arxiv.org/html/2609.01757v1",
    "sources/mimic-cxr-jpg.html": "https://physionet.org/content/mimic-cxr-jpg/2.1.0/",
    "sources/vindr-cxr.html": "https://physionet.org/content/vindr-cxr/1.0.0/",
    "sources/cxr-lt.html": "https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/",
    "sources/chexchonet.html": "https://physionet.org/content/chexchonet/1.0.0/",
    "sources/ms-cxr.html": "https://physionet.org/content/ms-cxr/1.1.0/",
    "sources/padchest.html": "https://bimcv.cipf.es/bimcv-projects/padchest/",
    "sources/padchest-gr.html": "https://bimcv.cipf.es/bimcv-projects/padchest-gr/",
    "sources/chexpert.html": "https://stanfordmlgroup.github.io/competitions/chexpert/",
    "sources/chexlocalize.html": "https://aimi.stanford.edu/datasets/chexlocalize",
    "sources/chexlocalize-README.md": "https://raw.githubusercontent.com/rajpurkarlab/cheXlocalize/master/README.md",
}


def fetch(item):
    name, url = item
    target = ASSETS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {"url": url, "path": str(target)}
    try:
        response = requests.get(url, timeout=(20, 120))
        result["http_status"] = response.status_code
        response.raise_for_status()
        if name.endswith(".pdf") and not response.content.startswith(b"%PDF"):
            raise ValueError("Response is not a PDF")
        target.write_bytes(response.content)
        result.update(bytes=target.stat().st_size, sha256=hashlib.sha256(response.content).hexdigest())
    except Exception as exc:
        result["error"] = str(exc)
    print(name, result.get("bytes", result.get("error")), flush=True)
    return name, result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        entries = dict(pool.map(fetch, URLS.items()))
    report = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "files": entries}
    (ASSETS / "manifests/source_pages.json").write_text(json.dumps(report, indent=2) + "\n")
