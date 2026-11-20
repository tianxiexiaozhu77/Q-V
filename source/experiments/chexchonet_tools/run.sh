#!/usr/bin/env bash
set -euo pipefail
chexcho_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chexcho_workspace="$(cd -- "$chexcho_tools/../.." && pwd)"
chexcho_output="${1:-$chexcho_workspace/experiments/chexchonet_comparison_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then shift; fi
exec python3 "$chexcho_tools/run_comparison.py" --output "$chexcho_output" "$@"
