#!/bin/bash
# Generic R9 Slurm wrapper. Usage:
#   sbatch -p cpu -J r9_<name> -c 8 --mem 64G -o assets/logs/%x_%j.out scripts/r9/slurm/run_py.sh scripts/r9/<script>.py [args...]
# For arrays the task id is exported as R9_TASK (SLURM_ARRAY_TASK_ID).
set -euo pipefail
source /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state/scripts/r9/env.sh
export R9_TASK="${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
PY="${R9_PY_OVERRIDE:-$R9_PY}"
echo "[r9] host=$(hostname) job=${SLURM_JOB_ID:-na} task=$R9_TASK commit=$(git -C "$R9_ROOT" rev-parse HEAD) start=$(date -Is)"
"$PY" "$@"
echo "[r9] end=$(date -Is)"
