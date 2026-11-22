#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=01:00:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
benchmark="$root/v0plus-runs/yang-external-54fcad5"
source_book="$root/v0plus-public-assets/yang-plc-rv2w3dv9rs-v2/mmc8.xlsx"
args=(--inputs "$benchmark/inputs")
case "$stage" in
  prepare)
    args+=(--teacher "$benchmark/final.parquet" --workbook "$source_book" --out "$run/splits")
    ;;
  predict)
    fold=$SLURM_ARRAY_TASK_ID
    args+=(
      --fold "$fold" --splits "$run/splits" --priors "$root/v0plus-public-priors"
      --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
      --targets "$root/external-code-v3/drug_targets.json"
      --out "$run/forecasts/fold-$fold.parquet"
    )
    ;;
  score)
    args+=(--teacher "$benchmark/final.parquet" --workbook "$source_book" --splits "$run/splits"
           --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.yang_hcc_adaptation "$stage" "${args[@]}"
