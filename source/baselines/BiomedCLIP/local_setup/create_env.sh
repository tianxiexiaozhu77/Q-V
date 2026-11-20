#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_ROOT="$(dirname -- "$BASELINE_ROOT")"
export UV_CACHE_DIR="$BASELINES_ROOT/.cache/uv" TMPDIR="$BASELINES_ROOT/.cache/tmp"
export UV_LOCK_TIMEOUT=7200 UV_HTTP_TIMEOUT=7200
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}pypi.org,files.pythonhosted.org"
export no_proxy="$NO_PROXY"
BASELINE_PYTHON="${BASELINE_PYTHON:-$BASELINES_ROOT/../AlphaRAD/local_assets/tools/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10}"
"$BASELINES_ROOT/.tools/bin/uv" venv --allow-existing --python "$BASELINE_PYTHON" "$BASELINE_ROOT/.venv"
"$BASELINES_ROOT/.tools/bin/uv" pip install --python "$BASELINE_ROOT/.venv/bin/python" -r "$BASELINE_ROOT/local_setup/requirements.lock.txt"
