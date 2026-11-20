#!/usr/bin/env bash
#SBATCH --job-name=rf_scgpt_fig3a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --output=logs/scgpt_fig3a_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
exec .venvs/scgpt/bin/python -u src/scgpt_reproduce.py --run-id R2-manuscript-public-v1 "$@"
