#!/usr/bin/env bash
# F1-diag: regenerate validation predictions from saved best checkpoints (no training, no test).
#SBATCH --job-name=f1_diag_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/f1_diag_infer_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
WT=$(pwd)
PRIOR=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_reproduction_first_20261105
OUT=$WT/assets/feedback/F1/diag
for rt in scgpt gears; do
  ".venvs/$rt/bin/python" -u src/feedback/f1_diag_infer.py --runtime "$rt" --prior-root "$PRIOR" \
    --dataset "$WT/prior_assets/common/C0-Norman-v1-s26" --runs "$WT/prior_assets/runs" \
    --public "$WT/prior_assets/public" --out "$OUT"
done
