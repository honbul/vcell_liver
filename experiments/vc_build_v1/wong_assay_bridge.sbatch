#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
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
case "$action" in
  prepare)
    args=(--original "$runs/wong-84d021d/cohort"
          --workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx" --out "$run/inputs")
    ;;
  predict)
    args=(--inputs "$run/inputs/predict" --out "$run/forecast.parquet")
    ;;
  score)
    args=(--inputs "$run/inputs" --forecast "$run/forecast.parquet"
          --pdo "$runs/wong-expanded-15b1345" --tissue "$runs/wong-tissue-a4120a9" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.wong_assay_bridge "$action" "${args[@]}"
