#!/usr/bin/env bash
set -euo pipefail
BASELINES_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_DIR="$BASELINES_DIR/runs/query_transfer_$(date +%Y%m%d_%H%M%S)_$$"
exec python "$BASELINES_DIR/local_setup/run_query_transfer.py" --output "$RUN_DIR" "$@"
