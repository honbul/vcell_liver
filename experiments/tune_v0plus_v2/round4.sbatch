#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
workspace=$2
runs=$3
out=$4
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
root="$study/vc_build_v1"
"$python" -m v0plus_v2.round4 --bundle "$root/function/cross-v1" --round2 "$runs/round2-02" \
  --round3 "$runs/round3-01" --fold "${SLURM_ARRAY_TASK_ID:?}" --out "$out/fold$SLURM_ARRAY_TASK_ID"
