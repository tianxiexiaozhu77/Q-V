#!/usr/bin/env bash
set -euo pipefail
qata_run_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qata_output_args=()
if [[ $# -gt 0 && "$1" != --* ]]; then
  qata_output_args=(--output "$1")
  shift
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec python "$qata_run_root/run_campaign.py" "${qata_output_args[@]}" "$@"
