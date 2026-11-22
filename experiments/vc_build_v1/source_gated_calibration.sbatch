#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=01:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
training=(--original "$root/function/cross-v1" --core "$root/v0plus-runs/source-core-92aa482/core")
archive=(--archive "$root/v0plus-runs/neural-interaction-f7f0dc5/forecasts"
         --archive-receipt "$repository/reports/vc_build_v1/v0plus/neural_interaction_forecast_receipts.json")
stamp=(--stamp "$run/calibrator_receipt.json")
calibrator=(--calibrator "$run/calibrator.json")
case "$stage" in
  fit)
    args=("${training[@]}" "${archive[@]}" "${stamp[@]}" --out "$run/calibrator.json")
    ;;
  predict)
    studies=(LIMORE LICOB)
    study=${studies[$SLURM_ARRAY_TASK_ID]}
    args=("${archive[@]}" "${stamp[@]}" "${calibrator[@]}" --study "$study" --out "$run/forecasts/$study.parquet")
    ;;
  score)
    args=("${training[@]}" "${calibrator[@]}" "${stamp[@]}"
           --forecasts "$run/forecasts" --licob "$root/v0plus-runs/licob-wide-rna-2833a20/cohort"
           --limore-labels "$root/function/cross-v1/test_labels/outcomes.parquet"
           --licob-labels "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
           --out "$run/external_score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.source_gated_calibration "$stage" "${args[@]}"
