#!/usr/bin/env bash
#SBATCH --job-name=rf_lora_check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --output=logs/common_lora_check_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
exec .venvs/scgpt/bin/python -u src/common_lora_check.py "$@"
