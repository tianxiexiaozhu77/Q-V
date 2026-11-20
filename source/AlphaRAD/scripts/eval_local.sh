#!/usr/bin/env bash
# Run released AlphaRAD; no training is performed.
set -euo pipefail
alpharad_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$alpharad_root"
source local_setup/activate.sh
alpharad_dataset="${1:-all}"
if (( $# > 0 )); then shift; fi
if (( $# == 0 )); then set -- 3 4 5 6 7; fi
case "$alpharad_dataset" in
  chestxray14) alpharad_dataset=chest14 ;;
  rsna) alpharad_dataset=rsna_pneumonia ;;
  siim) alpharad_dataset=siim_pneumothorax ;;
  qata) alpharad_dataset=qatacovid_seg ;;
esac
if [[ "$alpharad_dataset" == all ]]; then
  mapfile -t alpharad_datasets < <(python -c 'import json; print("\n".join(json.load(open("local_setup/pretrained_catalog.json"))["datasets"]))')
else
  alpharad_datasets=("$alpharad_dataset")
fi
if (( ${#alpharad_datasets[@]} == 0 )); then
  printf 'No prepared dataset manifests are available.\n' >&2
  exit 1
fi
alpharad_output="$alpharad_root/runs/pretrained_$(date +%Y%m%d_%H%M%S)_$$"
printf 'Results: %s/RESULTS.txt\n' "$alpharad_output"
exec python local_setup/run_pretrained.py \
  --catalog local_setup/pretrained_catalog.json \
  --datasets "${alpharad_datasets[@]}" --gpus "$@" --output "$alpharad_output"
