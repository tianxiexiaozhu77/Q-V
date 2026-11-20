#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_OUTPUT="${1:-$EXPERIMENT_ROOT/runs/full_$(date +%Y%m%d_%H%M%S)}"
exec python "$EXPERIMENT_ROOT/run_suite.py" \
  --output "$EXPERIMENT_OUTPUT" \
  --protocol-dir "$EXPERIMENT_ROOT/protocol_v1" \
  --models BiomedCLIP AlphaRAD RadZero \
  --datasets rsna_pneumonia chestxdet10 --gpus 4 5 6 7
