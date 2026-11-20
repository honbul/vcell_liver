#!/usr/bin/env bash
#SBATCH --job-name=rf_paper_diagnose
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=00:40:00
#SBATCH --output=logs/paper_diagnose_%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec .venvs/gears/bin/python -u src/paper_diagnose.py "$@"
