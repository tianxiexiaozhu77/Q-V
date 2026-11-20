#!/usr/bin/env bash
set -euo pipefail
stability_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Usage: bash experiments/chexchonet_stability_20261024/run.sh [NEW_OUTPUT_DIRECTORY]'
  echo 'BiomedCLIP + AlphaRAD; fixed 7407-image test; 5 support seeds; patient bootstrap; checked idle GPUs 4-7.'
  exit 0
fi
stability_output="${1:-$stability_root/runs/run_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python "$stability_root/run_campaign.py" --output "$stability_output" --gpus 4 5 6 7
