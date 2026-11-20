#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_OUTPUT="${1:-${TASK_ROOT}/runs/run_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then shift; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python "${TASK_ROOT}/run_campaign.py" --output "${RUN_OUTPUT}" "$@"
