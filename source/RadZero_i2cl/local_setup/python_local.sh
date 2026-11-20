#!/usr/bin/env bash
set -euo pipefail
RADZERO_I2CL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RADZERO_I2CL_ROOT/local_setup/activate.sh"
exec "$VIRTUAL_ENV/bin/python" "$@"
