#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:30:00
set -euo pipefail
study=$1
workspace=$2
run=$3
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
root="$study/vc_build_v1"
"$python" -m v0plus_v2.source --bundle "$root/function/cross-v1" \
  --targets "$root/external-code-v3/drug_targets.json" \
  --fold "${SLURM_ARRAY_TASK_ID:?}" --out "$run/fold$SLURM_ARRAY_TASK_ID"
