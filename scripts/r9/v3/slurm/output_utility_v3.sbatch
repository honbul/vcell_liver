#!/bin/bash
# P0-R9 v3 re-aggregation (CPU only, no training). Submit ONLY through the v3 wrapper:
#   /mnt/S1/sdata/workbench/Virtual_Cell/gssong/v3_tools/v3_sbatch.sh --job-name=v3r9_<name> --cpus-per-task=4 --mem=16G \
#     --time=02:00:00 -o assets/logs/%x_%j.out scripts/r9/v3/slurm/output_utility_v3.sbatch <step1|step2|all>
set -euo pipefail
export R9_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_output_utility_v3
source "$R9_ROOT/scripts/r9/v2/env_v2.sh"   # v1 r9core env (read-only), PYTHONPATH scripts/r9 + v2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
STEP="${1:-all}"
echo "[v3r9] host=$(hostname) job=${SLURM_JOB_ID:-na} step=$STEP commit=$(git -C "$R9_ROOT" rev-parse HEAD) dirty=$(git -C "$R9_ROOT" status --porcelain -uno | wc -l) start=$(date -Is)"
if [[ "$STEP" == step1 || "$STEP" == all ]]; then /usr/bin/time -v "$R9_PY" "$R9_ROOT/scripts/r9/v3/output_utility.py"; fi
if [[ "$STEP" == step2 || "$STEP" == all ]]; then /usr/bin/time -v "$R9_PY" "$R9_ROOT/scripts/r9/v3/output_utility_summary.py"; fi
echo "[v3r9] end=$(date -Is)"
