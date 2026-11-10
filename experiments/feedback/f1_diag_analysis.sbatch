#!/usr/bin/env bash
# F1-diag A1/A3/A4/A5 tables from saved artifacts and regenerated validation predictions (CPU).
#SBATCH --job-name=f1_diag_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/f1_diag_analysis_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
WT=$(pwd)
.venvs/scgpt/bin/python -u src/feedback/f1_diag_analysis.py \
  --dataset "$WT/prior_assets/common/C0-Norman-v1-s26" --runs "$WT/prior_assets/runs" \
  --study "$WT/prior_assets/study/C0-Norman-v1-s26" --diag "$WT/assets/feedback/F1/diag"
