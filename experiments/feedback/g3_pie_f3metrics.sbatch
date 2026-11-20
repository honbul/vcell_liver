#!/usr/bin/env bash
# G3 / G1-b part 2: F3 metrics (raw-data truth) for PIE + PIE-split baselines on the held-out line.
#SBATCH --job-name=g3_pie_f3m
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/g3_pie_f3metrics_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
# read-only use of the existing F3 env (numpy/scipy/sklearn); nothing is installed into it
.venvs/scgpt/bin/python -u src/feedback/g3_pie_f3metrics.py --fold "${1:-hepg2}" \
  --g3 assets/g3_pie/baselines --f3 assets/f3 --out reports/feedback/G3
echo "end=$(date -Is)"
