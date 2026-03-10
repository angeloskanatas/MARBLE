#!/bin/bash
# Run HXMSA extraction + probing for all models.
#
# Usage:
#   bash scripts/run_all_hxmsa.sh              # run all models
#   bash scripts/run_all_hxmsa.sh MuQ MusicFM  # run specific models only
set -e
cd "$(dirname "$0")/.."

ALL_MODELS=(MuQ MusicFM MERT-v1-95M MERT-v1-330M OMAR-RQ CLAP MAEST MusicGen MusicFlamingo Qwen2Audio)

if [ $# -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${ALL_MODELS[@]}")
fi

for MODEL in "${MODELS[@]}"; do
  echo "========================================"
  echo "MODEL: $MODEL"
  echo "========================================"

  # Step 1: Extract embeddings
  echo "--- Extracting embeddings ---"
  bash scripts/extract.${MODEL}.HXMSA.sh

  # Step 2: Probe layerwise
  echo "--- Probing HXMSA (layerwise) ---"
  bash scripts/probe.${MODEL}.HXMSA.layerwise.sh

  echo "MODEL $MODEL done."
  echo ""
done

echo "All done!"
