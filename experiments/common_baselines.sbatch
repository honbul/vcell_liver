#!/usr/bin/env bash
#SBATCH --job-name=rf_baselines
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/common_baselines_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
exec .venvs/gears/bin/python -u src/common_baselines.py "$@"
