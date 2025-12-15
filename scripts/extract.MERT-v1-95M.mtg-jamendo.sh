#!/bin/bash
#SBATCH --job-name=extract-mert-95m-mtgjamendo
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/akanatas/MARBLE
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate layerbylayer

module load CUDA/11.4.3

BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_SAMPLES=${MAX_SAMPLES:-50000}
MAX_FILES=${MAX_FILES:-null}
NUM_AUGMENTATIONS=${NUM_AUGMENTATIONS:-2}
SPLIT=${SPLIT:-test}
echo "BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS MAX_SAMPLES=$MAX_SAMPLES MAX_FILES=$MAX_FILES NUM_AUGMENTATIONS=$NUM_AUGMENTATIONS SPLIT=$SPLIT"

ARGS="--data.init_args.batch_size=${BATCH_SIZE} --data.init_args.num_workers=${NUM_WORKERS}"

if [ "$MAX_SAMPLES" != "null" ]; then
    ARGS="$ARGS --model.init_args.extraction.max_samples=${MAX_SAMPLES}"
fi

if [ "$MAX_FILES" != "null" ]; then
    ARGS="$ARGS --data.init_args.${SPLIT}.init_args.max_files=${MAX_FILES}"
fi

if [ "$NUM_AUGMENTATIONS" != "2" ]; then
    ARGS="$ARGS --model.init_args.extraction.augmentation.num_augmentations=${NUM_AUGMENTATIONS}"
fi

if [ "$SPLIT" != "test" ]; then
    ARGS="$ARGS --model.init_args.extraction.split=${SPLIT}"
fi

python cli.py test -c configs/extract.MERT-v1-95M.mtg-jamendo.yaml $ARGS
