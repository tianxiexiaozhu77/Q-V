#!/usr/bin/env bash
set -euo pipefail
alpharad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
medical_root="$(dirname -- "$alpharad_root")"
assets="$alpharad_root/local_assets"
uv_bin="$assets/tools/uv"
export UV_CACHE_DIR="$assets/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$assets/tools/python"
export UV_PYTHON_BIN_DIR="$assets/tools/python-bin"
export TMPDIR="$assets/.cache/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR"
if [[ ! -x "$uv_bin" ]]; then
    echo "Missing project-local uv: $uv_bin" >&2
    exit 1
fi
if [[ ! -x "$alpharad_root/.venv/bin/python" ]]; then
    "$uv_bin" python install 3.10.22
    "$uv_bin" venv --python 3.10.22 "$alpharad_root/.venv"
fi
"$uv_bin" pip install --python "$alpharad_root/.venv/bin/python" setuptools==65.6.3 wheel==0.38.4
"$uv_bin" pip install --python "$alpharad_root/.venv/bin/python" --no-build-isolation -r "$alpharad_root/local_setup/runtime-resolved.txt"
"$alpharad_root/.venv/bin/python" -m pip check
"$alpharad_root/.venv/bin/python" -m pip freeze --all > "$alpharad_root/local_setup/runtime.lock.txt"
