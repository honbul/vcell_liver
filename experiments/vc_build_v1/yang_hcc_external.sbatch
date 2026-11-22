#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:00:00
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
case "$stage" in
  prepare)
    args=(
      --registration "$root/v0plus-runs/yang-hcc-445841e/cohort"
      --original "$root/function/cross-v1"
      --core "$root/v0plus-runs/source-core-92aa482/core"
      --extension "$root/v0plus-runs/source-panel-c351b0b/extension"
      --curves /mnt/S1/sdata/pubdata/DepMap/DrugScreens/PRISM_Repurposing_20Q2/prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv
      --broad "$root/v0plus-runs/wong-84d021d/source/repurposing_samples_20200324.txt"
      --out "$run/inputs"
    )
    ;;
  cpu|gpu)
    if [[ "$stage" == cpu ]]; then
      arms=(linear lightgbm cellhit)
      device=()
    else
      arms=(catboost velodrome_inductive codeae_inductive v0plus_inductive structured)
      device=(--gpu)
    fi
    arm=${arms[$SLURM_ARRAY_TASK_ID]}
    stage=predict
    args=(
      --inputs "$run/inputs" --arm "$arm"
      --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
      --priors "$root/v0plus-public-priors"
      --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
      --targets "$root/external-code-v3/drug_targets.json"
      --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
      --upstream "$root/v0plus-upstream"
      --out "$run/predictions/$arm.parquet" "${device[@]}"
    )
    ;;
  merge)
    args=(--inputs "$run/inputs" --parts "$run/predictions" --out "$run/final.parquet")
    ;;
  score)
    args=(
      --inputs "$run/inputs" --forecast "$run/final.parquet"
      --workbook "$root/v0plus-public-assets/yang-plc-rv2w3dv9rs-v2/mmc8.xlsx"
      --out "$run/score.json"
    )
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.yang_hcc_external "$stage" "${args[@]}"
