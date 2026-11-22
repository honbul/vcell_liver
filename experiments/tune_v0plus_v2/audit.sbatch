#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=01:45:00
set -euo pipefail
study=$1
workspace=$2
out=$3
stage=$4
cd "$workspace"
patient="$study/hcc_patient_eval_v1"
export PYTHONPATH="$workspace:$workspace/src:$study/vc_build_v1/v0plus-model-deps:$study/liver_dili_v1/deps:$patient/code:$patient/deps"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m v0plus_v2.audit "$stage" --v1-runs "$study/vc_build_v1/v0plus-runs" \
  --package "$out/package" --patient "$patient" --out "$out/audits"
