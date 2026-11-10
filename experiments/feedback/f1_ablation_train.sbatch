#!/usr/bin/env bash
#SBATCH --job-name=f1_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=07:00:00
#SBATCH --output=logs/f1_ablation_%j.log
# One registered F1 ablation run (validation only). Usage:
#   sbatch experiments/feedback/f1_ablation_train.sbatch --run-id <A-...> [--resume]
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID:-} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
exec .venvs/scgpt/bin/python -u src/feedback/f1_ablation_train.py "$@"
