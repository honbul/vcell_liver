#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

stage="$1"
registered="$2"
base="$(dirname "$(dirname "$stage")")"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''

exec "$python" -m src.r19_plan.r18_preflight --registered "$registered" --base "$base" \
    --reference "$stage/reports/r19_plan/data/r18-support-receipt.json" --out "$stage/r18-input-lock"
