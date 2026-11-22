#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=26G
#SBATCH --time=02:30:00
set -euo pipefail
study=$1
workspace=$2
run=$3
round1=$4
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps:$study/liver_dili_v1/deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
root="$study/vc_build_v1"
"$python" -m v0plus_v2.round2 --bundle "$root/function/cross-v1" \
  --targets "$root/external-code-v3/drug_targets.json" --round1 "$round1" \
  --fold "${SLURM_ARRAY_TASK_ID:?}" --out "$run/fold$SLURM_ARRAY_TASK_ID"
