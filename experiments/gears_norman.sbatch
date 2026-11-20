#!/usr/bin/env bash
#SBATCH --job-name=rf_gears_fig2f
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=1-5%2
#SBATCH --output=logs/gears_fig2f_%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
seed="${SLURM_ARRAY_TASK_ID:?}"
exec .venvs/gears/bin/python -u src/gears_reproduce.py --split-seed "$seed" --run-id "R1-paper-public-v1-s$seed" "$@"
