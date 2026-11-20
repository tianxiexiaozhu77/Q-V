#!/usr/bin/env bash
set -euo pipefail
VARIANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_ROOT="$(cd -- "$VARIANT_ROOT/../../.." && pwd)"
BASE_MODEL_ROOT="$BASELINES_ROOT/BioViL-T"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$BASELINES_ROOT/.cache/huggingface" XDG_CACHE_HOME="$BASELINES_ROOT/.cache" TMPDIR="$BASELINES_ROOT/.cache/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$VARIANT_ROOT/local_setup:$BASE_MODEL_ROOT/official${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "$BASELINES_ROOT/local_setup/evaluate_baseline.py" ]]; then
  shift
  set -- "$BASELINES_ROOT/local_setup/evaluate_baseline_buffered.py" "$@"
fi
exec "$BASE_MODEL_ROOT/.venv/bin/python" "$@"
