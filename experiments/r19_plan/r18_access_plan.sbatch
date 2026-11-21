#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail

stage="$1"
lock="$2"
registry="$3"
index="$4"
base="$(dirname "$(dirname "$stage")")"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''
preflight="$stage/reports/r19_plan/data/preflight/r18"

exec "$python" -m src.r19_plan.r18_access_plan \
    --lock "$lock" --lock-reference "$preflight/receipt.json" \
    --registered "$registry" --registered-reference "$stage/reports/r19_plan/data/r18-support-receipt.json" \
    --index "$index" --index-reference "$preflight/index" --out "$stage/allowlists"
