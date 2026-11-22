#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
args=()
case "$stage" in
  cv|external-predict)
    if [[ "$stage" == cv ]]; then
      assays=(pooled CTRPv2 GDSC2 PRISM20Q2)
      mode=${assays[$((SLURM_ARRAY_TASK_ID / 3))]}
      fold=$((SLURM_ARRAY_TASK_ID % 3))
      outfile="$mode-$fold.parquet"
    else
      studies=(LIMORE LICOB)
      mode=${studies[$SLURM_ARRAY_TASK_ID]}
      fold=0
      outfile="$mode.parquet"
    fi
    stage=predict
    args+=(--mode "$mode" --fold "$fold" --archive "$root/v0plus-runs/neural-interaction-f7f0dc5/forecasts"
           --receipt "$repository/reports/vc_build_v1/v0plus/neural_interaction_forecast_receipts.json"
           --out "$run/forecasts/$outfile")
    ;;
  source|external)
    args+=(--original "$root/function/cross-v1" --core "$root/v0plus-runs/source-core-92aa482/core"
           --forecasts "$run/forecasts" --out "$run/${stage}_score.json")
    if [[ "$stage" == external ]]; then
      args+=(--licob "$root/v0plus-runs/licob-wide-rna-2833a20/cohort"
             --limore-labels "$root/function/cross-v1/test_labels/outcomes.parquet"
             --licob-labels "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx")
    fi
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.neural_offset_factorial "$stage" "${args[@]}"
