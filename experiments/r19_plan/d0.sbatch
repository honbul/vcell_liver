#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

stage="$1"
base="$(dirname "$(dirname "$stage")")"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$stage:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=''

exec "$python" -m src.r19_plan.reconcile_r17 --original "$base/vcell_liver_r17_lincs/v1" \
    --cache "$base/vcell_liver_r15_lincs/v1/cache" --lincs /mnt/S1/sdata/pubdata/LINCS \
    --reference "$stage/reports/r17_lincs/data" --out "$stage/d0-r17-reconciliation" \
    --protocol "$stage/reports/r17_lincs/PROTOCOL.md" --r15-protocol "$stage/reports/r15_lincs/PROTOCOL.md"
