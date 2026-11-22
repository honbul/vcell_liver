#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=25G
#SBATCH --time=03:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
mode=$4
shift 4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
bundle="$root/function/cross-v1"

case "$mode" in
  register)
    args=(--workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
          --broad "$run/source/repurposing_samples_20200324.txt" --cas "$run/source/cas"
          --bundle "$bundle" --out "$run/cohort")
    ;;
  predict)
    arm=$1
    shift
    args=(--cohort "$run/cohort" --bundle "$bundle"
          --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --upstream "$root/v0plus-upstream"
          --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
          --arm "$arm" --out "$run/predictions/$arm.parquet" "$@")
    ;;
  merge)
    args=(--cohort "$run/cohort" --parts "$run/predictions" --out "$run/forecast.parquet")
    ;;
  score)
    args=(--cohort "$run/cohort" --forecast "$run/forecast.parquet"
          --workbook "$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.v0plus_wong "$mode" "${args[@]}"
