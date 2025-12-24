#!/bin/bash
#SBATCH --job-name=extract_mert-95M_discotube
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:quadro:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/akanatas/projects/layerbylayer/MARBLE
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate layerbylayer

module load CUDA/11.4.3

python cli.py test -c configs/extract.MERT-v1-95M.discotube.yaml
