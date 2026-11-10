#!/usr/bin/env bash
# G4: F3 metrics for State zeroshot (4 folds) and/or PIE seed retrains (HepG2). CPU, read-only F3 env.
# Usage: sbatch experiments/feedback/g4_f3metrics.sbatch [state|pie_seeds|both]
#SBATCH --job-name=g4_f3m
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/g4_f3metrics_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
.venvs/scgpt/bin/python -u src/feedback/g4_f3metrics.py --part "${1:-both}" --out reports/feedback/G4
echo "end=$(date -Is)"
