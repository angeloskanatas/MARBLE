#!/bin/bash
#SBATCH --job-name=extract_muq_gtzan_genre
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/akanatas/projects/layerbylayer/MARBLE
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate layerbylayer

module load CUDA/11.4.3

set -e
OUT=./output/extracted_embeddings_probing/muq_gtzan_genre
for split in train val test; do
  echo "Extracting split: $split"
  python cli.py test -c configs/extract.MuQ.GTZANGenre.yaml --model.init_args.extraction.split "$split"
done
echo "Done. Embeddings in $OUT/{train,val,test}/layer{N}/sequence-level/"
