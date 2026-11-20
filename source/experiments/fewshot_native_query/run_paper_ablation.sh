#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf '%s\n' \
    'Usage: bash experiments/fewshot_native_query/run_paper_ablation.sh [NEW_OUTPUT_DIRECTORY]' \
    'Run BiomedCLIP and AlphaRAD on RSNA and ChestXDet10 using checked idle GPUs 4-7.' \
    'The output directory must not exist; the default uses a timestamp.'
  exit 0
fi
if [[ $# -gt 1 ]]; then
  printf '%s\n' 'Expected at most one argument: a new output directory. See --help.' >&2
  exit 2
fi

ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_OUTPUT="${1:-$ABLATION_ROOT/runs/paper_ablation_$(date +%Y%m%d_%H%M%S)}"
exec python "$ABLATION_ROOT/run_suite.py" \
  --output "$ABLATION_OUTPUT" \
  --protocol-dir "$ABLATION_ROOT/protocol_v1" \
  --models BiomedCLIP AlphaRAD \
  --datasets rsna_pneumonia chestxdet10 \
  --gpus 4 5 6 7
