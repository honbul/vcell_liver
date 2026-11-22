#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

stage="$1"
mode="$2"
base="$(dirname "$(dirname "$stage")")"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$stage:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=''

if [ "$mode" = "phenotypes" ]; then
    exec "$python" -m src.r19_plan.i_kernel_audit --root "$stage" --coverage "$stage/coverage-presence" \
        --views I2 I4 --out "$stage/kernel-phenotypes"
fi
exec "$python" -m src.r19_plan.i_kernel_audit --root "$stage" --out "$stage/kernel-audit"
