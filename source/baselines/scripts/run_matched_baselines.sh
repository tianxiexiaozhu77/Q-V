#!/usr/bin/env bash
set -euo pipefail
baseline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
baseline_output="$baseline_root/runs/matched_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
exec bash "$baseline_root/../AlphaRAD/local_setup/python_local.sh" \
  "$baseline_root/local_setup/run_matched_baselines.py" --output "$baseline_output" "$@"
