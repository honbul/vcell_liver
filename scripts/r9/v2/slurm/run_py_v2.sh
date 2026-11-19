#!/bin/bash
# Generic R9 v2 Slurm wrapper (CPU). Usage:
#   sbatch -p cpu -J r9v2_<name> -c 8 --mem 32G -o assets/logs/%x_%A_%a.out [-a 0-N%T] scripts/r9/v2/slurm/run_py_v2.sh <script.py> [args...]
set -euo pipefail
source /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_norm_proto_v2/scripts/r9/v2/env_v2.sh
export R9_TASK="${R9_TASK:-${SLURM_ARRAY_TASK_ID:-0}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "[r9v2] host=$(hostname) job=${SLURM_JOB_ID:-na} task=$R9_TASK commit=$(git -C "$R9_ROOT" rev-parse HEAD) start=$(date -Is)"
"$R9_PY" "$@"
echo "[r9v2] end=$(date -Is)"
