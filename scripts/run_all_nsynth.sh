#!/bin/bash
# Run NSynth extraction + probing (NSynthI & NSynthP) for all models.
# Extraction is shared: same embeddings used for both instrument and pitch probes.
#
# Usage:
#   bash scripts/run_all_nsynth.sh              # run all models
#   bash scripts/run_all_nsynth.sh MuQ MusicFM  # run specific models only
set -e
cd "$(dirname "$0")/.."

ALL_MODELS=(MuQ MusicFM MERT-v1-95M MERT-v1-330M  OMAR-RQ CLAP MAEST MusicGen MusicFlamingo Qwen2Audio)

if [ $# -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${ALL_MODELS[@]}")
fi

for MODEL in "${MODELS[@]}"; do
  echo "========================================"
  echo "MODEL: $MODEL"
  echo "========================================"

  # Step 1: Extract embeddings (shared for NSynthI & NSynthP)
  echo "--- Extracting embeddings ---"
  bash scripts/extract.${MODEL}.NSynth.sh

  # Step 2: Probe NSynthI (instrument family, 11 classes)
  echo "--- Probing NSynthI ---"
  bash scripts/probe.${MODEL}.NSynthI.layerwise.sh

  # Step 3: Probe NSynthP (pitch, 128 classes)
  echo "--- Probing NSynthP ---"
  bash scripts/probe.${MODEL}.NSynthP.layerwise.sh

  echo "MODEL $MODEL done."
  echo ""
done

echo "All done!"
