#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --time=00:55:00
set -euo pipefail
study=$1
code=$2
runs=$3
stage=$4
cd "$code"
export PYTHONPATH="$code:$study/liver_dili_v1/deps:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r4="$runs/round4-01"
case "$stage" in
  lock) "$python" -m liver_dili_v2.round4 lock --round4 "$r4" --out "$r4/policy.json" ;;
  develop)
    case "${SLURM_ARRAY_TASK_ID:?}" in
      0|1|2|3|4) kind=murcko; fold=$SLURM_ARRAY_TASK_ID ;;
      *) kind=butina; fold=$((SLURM_ARRAY_TASK_ID - 5)) ;;
    esac
    "$python" -m liver_dili_v2.round4 develop --round3 "$runs/round3-02" \
      --kind "$kind" --fold "$fold" --out "$r4/$kind-$fold"
    ;;
  *) exit 2 ;;
esac
