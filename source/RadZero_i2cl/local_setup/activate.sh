#!/usr/bin/env bash
# Source this file to activate the project environment and keep caches local.
RADZERO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RADZERO_ASSETS="${RADZERO_ASSETS:-$RADZERO_ROOT/local_assets}"
export RADZERO_ROOT RADZERO_ASSETS
export HF_HOME="$RADZERO_ASSETS/.cache/huggingface"
export TORCH_HOME="$RADZERO_ASSETS/.cache/torch"
export XDG_CACHE_HOME="$RADZERO_ASSETS/.cache"
export MPLCONFIGDIR="$RADZERO_ASSETS/.cache/matplotlib"
export UV_CACHE_DIR="$RADZERO_ROOT/../.cache/uv"
export UV_PYTHON_INSTALL_DIR="$RADZERO_ROOT/../.tools/python"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_MODULES_CACHE="$RADZERO_ASSETS/.cache/huggingface/modules"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$RADZERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "$RADZERO_ROOT/.venv/bin/activate" ]]; then
  source "$RADZERO_ROOT/.venv/bin/activate"
elif [[ -f "$RADZERO_ROOT/../.envs/radzero/bin/activate" ]]; then
  source "$RADZERO_ROOT/../.envs/radzero/bin/activate"
else
  echo "RadZero environment is missing: $RADZERO_ROOT/../.envs/radzero" >&2
  return 1
fi
