#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
cd "$RADZERO_ROOT"
if [[ ! -d "$RADZERO_ASSETS/data/MIMIC-CXR/images" ]]; then
    printf '%s\n' "MIMIC-CXR images are not installed. See SETUP_LOCAL.md before training." >&2
    exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    exp/cxr_pt/run.py --add_cfg_list radzero paths_local \
    --no_report --inference False "$@"
