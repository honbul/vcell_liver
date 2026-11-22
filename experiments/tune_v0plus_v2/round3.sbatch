#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
workspace=$2
run=$3
cd "$workspace"
export PYTHONPATH="$workspace:$workspace/src:$study/vcell_liver/vcell_r12_runtime_cu126:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=5 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
root="$study/vc_build_v1"
"$python" -m v0plus_v2.round3 --bundle "$root/function/cross-v1" \
  --fm "$study/nextgen_5track/fm_store_v1" --cmap "$study/top3_tuning_v2/a/input/cmap24-01" --fold "${SLURM_ARRAY_TASK_ID:?}" --out "$run/fold$SLURM_ARRAY_TASK_ID"
