#!/usr/bin/env bash
# F3 QC: consistency check of our deltas/DEGs vs HKU CRISPRi12 deposit (md5-verified first)
#SBATCH --job-name=f3_xchk
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/f3_crosscheck_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
.venvs/scgpt/bin/python -u src/feedback/f3_crosscheck_hku.py --assets assets/f3 --out reports/feedback/F3
echo "end=$(date -Is)"
