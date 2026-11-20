#!/usr/bin/env bash
# Original AlphaRAD and fixed phase6 query refinement + two views on CheXpert.
set -euo pipefail
alpharad_chexpert_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$alpharad_chexpert_root"
alpharad_chexpert_output="$alpharad_chexpert_root/runs/chexpert_phase6_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
exec bash "$alpharad_chexpert_root/local_setup/python_local.sh" \
  "$alpharad_chexpert_root/local_setup/run_chexpert_query_comparison.py" \
  --output "$alpharad_chexpert_output" "$@"
