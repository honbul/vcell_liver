#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:50:00
set -euo pipefail

repository=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
bundle="$root/function/cross-v1"

case "$action" in
  predict)
    assays=(pooled CTRPv2 GDSC2 PRISM20Q2)
    assay=${assays[$((SLURM_ARRAY_TASK_ID / 3))]}
    fold=$((SLURM_ARRAY_TASK_ID % 3))
    args=(--bundle "$bundle" --baseline "$root/v0plus-runs/structured-bf73a0a"
          --assay "$assay" --fold "$fold" --out "$run/forecasts/$assay-$fold.parquet")
    ;;
  score)
    args=(--bundle "$bundle" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.anchor_rank_cv "$action" "${args[@]}"
