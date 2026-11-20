#!/usr/bin/env bash
# Rebuild this baseline's isolated inference environment from the validated lock.
set -euo pipefail
baseline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baselines_root="$(dirname -- "$baseline_root")"
medical_root="$(dirname -- "$baselines_root")"
export UV_CACHE_DIR="$baselines_root/.cache/uv"
export PIP_CACHE_DIR="$baselines_root/.cache/pip"
export TMPDIR="$baselines_root/.cache/tmp"
export UV_HTTP_TIMEOUT=7200 UV_LOCK_TIMEOUT=7200
export NO_PROXY="${NO_PROXY:-},pypi.org,files.pythonhosted.org"
export no_proxy="${no_proxy:-},pypi.org,files.pythonhosted.org"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
uv_bin="${BASELINE_UV:-$baselines_root/.tools/bin/uv}"
python_bin="${BASELINE_PYTHON:-$medical_root/AlphaRAD/local_assets/tools/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10}"
if [[ ! -x "$uv_bin" ]]; then
 echo "uv is missing: install uv or set BASELINE_UV to its executable." >&2; exit 1
fi
if [[ ! -x "$python_bin" ]]; then
 echo "Python 3.10 is missing: set BASELINE_PYTHON to a Python 3.10 executable." >&2; exit 1
fi
"$uv_bin" venv --python "$python_bin" "$baseline_root/.venv" --allow-existing
"$uv_bin" pip sync --python "$baseline_root/.venv/bin/python" "$baseline_root/local_setup/requirements.lock.txt"
"$baseline_root/.venv/bin/python" -c 'import sys,torch,torchvision,transformers; assert sys.prefix != sys.base_prefix; print(sys.executable, torch.__version__, torchvision.__version__, transformers.__version__)'
