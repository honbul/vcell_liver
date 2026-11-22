#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail

stage="$1"
base="$(dirname "$(dirname "$stage")")"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''

exec "$python" -m src.r19_plan.d1_metadata \
    --assignments "$base/vcell_liver_r17_lincs/v1/support/assignments.tsv" \
    --reference "$stage/reports/r17_lincs/data/support/receipt.json" --out "$stage/d1-control-lock"
