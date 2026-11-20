#!/usr/bin/env bash
# F3 step 2: per-line pseudobulk sufficient statistics (array 0-3 = K562, RPE1, HepG2, Jurkat)
#SBATCH --job-name=f3_pbulk
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/f3_pseudobulk_%A_%a.log
set -euo pipefail
LINES=(K562 RPE1 HepG2 Jurkat)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
.venvs/scgpt/bin/python -u src/feedback/f3_pseudobulk.py --line "${LINES[$SLURM_ARRAY_TASK_ID]}" --out assets/f3/pseudobulk
echo "end=$(date -Is)"
