#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --array=0-2
set -euo pipefail

stage="$1"
lock="$2"
registry="$3"
studies=(GSE92742 GSE70138 CMap2020)
study="${studies[$SLURM_ARRAY_TASK_ID]}"
base="$(dirname "$(dirname "$stage")")"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''

exec "$python" -m src.r19_plan.r18_matrix_index --study "$study" --base "$base" \
    --lock "$lock" --lock-reference "$stage/reports/r19_plan/data/preflight/r18/receipt.json" \
    --registered "$registry" --registered-reference "$stage/reports/r19_plan/data/r18-support-receipt.json" \
    --r17-reference "$stage/reports/r19_plan/data/r17-reconciliation.json" --out "$stage/index/$study"
