#!/usr/bin/env bash
#SBATCH --job-name=rf_common_pipeline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/common_pipeline_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec .venvs/gears/bin/python -u src/common_pipeline.py "$@"
