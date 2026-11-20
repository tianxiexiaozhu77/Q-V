#!/usr/bin/env bash
# Default: all nine datasets / ten tasks, fixed eta 0.3, checked idle GPUs 3–7.
# Example: bash scripts/eval_query_refinement.sh --datasets chest14 --gpus 3 --eta 0.3
# All arguments are forwarded; --output can override the fresh default directory.
set -euo pipefail
alpharad_query_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$alpharad_query_root"
source local_setup/activate.sh
alpharad_query_output="$alpharad_query_root/runs/query_refinement_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
exec python local_setup/run_query_refinement.py --output "$alpharad_query_output" "$@"
