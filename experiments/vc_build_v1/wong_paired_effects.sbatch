#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=00:45:00
set -euo pipefail
repository=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
fixed="$runs/wong-assay-bridge-a4411e1/inputs"
case "$action" in
  predict)
    args=(--inputs "$fixed/predict" --tissue "$runs/wong-tissue-a4120a9/final.parquet" --out "$run/forecast.parquet")
    ;;
  score)
    args=(--inputs "$fixed" --forecast "$run/forecast.parquet" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.wong_paired_effects "$action" "${args[@]}"
