#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_output="${1:-$task_dir/runs/run_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then shift; fi
exec python "$task_dir/run_campaign.py" --output "$task_output" --gpus 6 7 "$@"
