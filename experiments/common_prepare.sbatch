#!/usr/bin/env bash
#SBATCH --job-name=rf_c0_data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=logs/common_prepare_%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
exec .venvs/gears/bin/python -u src/common_data.py "$@"
