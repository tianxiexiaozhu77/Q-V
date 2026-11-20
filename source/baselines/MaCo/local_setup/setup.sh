#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$root/../.cache/uv" TMPDIR="$root/../.cache/tmp"
export UV_LOCK_TIMEOUT=3600 UV_HTTP_TIMEOUT=120
export NO_PROXY="${NO_PROXY:-},pypi.org,files.pythonhosted.org"
export no_proxy="$NO_PROXY"
uv="$root/../.tools/bin/uv"
"$uv" venv --allow-existing --python /usr/bin/python3.10 "$root/.venv"
"$uv" pip install --python "$root/.venv/bin/python" 'torch==2.2.2' 'torchvision==0.17.2' 'gdown==6.4.2' 'setuptools<81' -r "$root/local_setup/requirements_eval.txt"
if [[ ! -f "$root/checkpoints/maco.pth" ]]; then
 "$root/.venv/bin/python" "$root/local_setup/download_ranges.py" 'https://drive.usercontent.google.com/download?id=1B87JuLJnjh_CoQpjT7RWGTp6alfXV8_0&export=download&confirm=t' "$root/checkpoints/maco.pth"
fi
