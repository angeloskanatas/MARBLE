#!/bin/bash
#
# Usage: sbatch run_task_all_models.sh TASK [extract|probe|both]
#   TASK e.g. GTZANGenre, GTZANBeatTracking, MTT, MTGGenre, GS, EMO, ...
#   Mode: extract | probe | both (default: both = extract then probe per model, sequentially)
#
#
#SBATCH --job-name=mtg_genre_all
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/akanatas/projects/layerbylayer/MARBLE
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

TASK="${1:?Usage: $0 TASK [extract|probe|both]}"
MODE="${2:-both}"

ONLY_MODELS=()

if [[ "$MODE" != "extract" && "$MODE" != "probe" && "$MODE" != "both" ]]; then
  echo "MODE must be extract, probe, or both (default)."
  exit 1
fi

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate layerbylayer
module load CUDA/11.4.3
export TOKENIZERS_PARALLELISM=false

get_models() {
  local task="$1"
  for f in configs/extract.*."$task".yaml; do
    [[ -f "$f" ]] || continue
    base=$(basename "$f" .yaml)
    model="${base#extract.}"
    model="${model%.$task}"
    [[ -f "scripts/extract.$model.$task.sh" ]] || continue
    [[ -f "configs/probe.$model.$task.layerwise.yaml" ]] || continue
    [[ -f "scripts/probe.$model.$task.layerwise.sh" ]] || continue
    echo "$model"
  done | sort -u
}

if [[ ${#ONLY_MODELS[@]} -gt 0 ]]; then
  MODELS=("${ONLY_MODELS[@]}")
else
  MODELS=($(get_models "$TASK"))
fi
OTHER=()
for m in "${MODELS[@]}"; do
  [[ "$m" == "MAEST" || "$m" == "MusicFlamingo" ]] || OTHER+=("$m")
done
LAST=()
[[ " ${MODELS[*]} " == *" MAEST "* ]] && LAST+=("MAEST")
[[ " ${MODELS[*]} " == *" MusicFlamingo "* ]] && LAST+=("MusicFlamingo")
MODELS=("${OTHER[@]}" "${LAST[@]}")

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models found for task=$TASK (need extract.*.$TASK.sh, probe.*.$TASK.layerwise.yaml + .sh)"
  exit 1
fi

echo "Task: $TASK  Mode: $MODE  Models (${#MODELS[@]}): ${MODELS[*]}"
echo "---"

for model in "${MODELS[@]}"; do
  echo "========== Model: $model =========="

  if [[ "$MODE" == "extract" || "$MODE" == "both" ]]; then
    echo "--- Extraction: $model / $TASK ---"
    bash "scripts/extract.$model.$TASK.sh"
    echo "  Extraction done."
  fi

  if [[ "$MODE" == "probe" || "$MODE" == "both" ]]; then
    echo "--- Probing: $model / $TASK ---"
    bash "scripts/probe.$model.$TASK.layerwise.sh"
    echo "  Probing done."
  fi

  echo ""
done

echo "========== All done for task=$TASK =========="
