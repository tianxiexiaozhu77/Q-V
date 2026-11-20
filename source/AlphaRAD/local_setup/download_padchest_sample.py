"""Download the small public sample linked by PadChest's official website."""
import importlib.util
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "alpharad-assets"
helper = ROOT.parent / "radzero-assets/data_images/download_public.py"
spec = importlib.util.spec_from_file_location("public_archive", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
url = "https://www.kaggle.com/api/v1/datasets/download/raddar/padchest-chest-xrays-sample"
with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
    if response.status != 206:
        raise RuntimeError("Expected a byte range response")
    size = int(response.headers["Content-Range"].split("/")[-1])
    response.read()
module.ROOT = ASSETS / "downloads/root"
module.DATASETS = {
    "PadChestSample": {
        "url": url,
        "bytes": size,
        "source_page": "https://www.kaggle.com/datasets/raddar/padchest-chest-xrays-sample",
        "official_referral": "https://bimcv.cipf.es/bimcv-projects/padchest/",
        "destination": str(ASSETS / "data_images/PadChestSample"),
        "scope": "Public sample only; not the full 39053-image AlphaRAD evaluation set.",
    }
}
module.download("PadChestSample")
record = json.loads((module.ROOT / "PadChestSample_source.json").read_text())
(ASSETS / "manifests/padchest_sample.json").write_text(json.dumps(record, indent=2) + "\n")
