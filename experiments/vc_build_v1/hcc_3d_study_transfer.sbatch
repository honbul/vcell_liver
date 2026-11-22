#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
stage=$4
shift 4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
benchmark="$root/v0plus-runs/yang-external-54fcad5"
licob="$root/v0plus-runs/licob-author-2f1328b"
forward=(
  --inputs "$run/inputs"
  --priors "$root/v0plus-public-priors"
  --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
  --targets "$root/external-code-v3/drug_targets.json"
)
case "$stage" in
  prepare)
    args=(
      --benchmark "$benchmark/inputs"
      --yang-workbook "$root/v0plus-public-assets/yang-plc-rv2w3dv9rs-v2/mmc8.xlsx"
      --licob "$licob/cohort" --patient-audit "$licob/patient-audit.json"
      --wong "$root/v0plus-runs/wong-84d021d/cohort"
      --out "$run/inputs"
    )
    ;;
  source)
    component=$1
    args=("${forward[@]}" --component "$component"
          --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
          --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
          --upstream "$root/v0plus-upstream" --out "$run/predictions/$component.parquet")
    ;;
  recipient)
    studies=(LICOB Wong)
    study=${studies[$SLURM_ARRAY_TASK_ID]}
    args=("${forward[@]}" --study "$study" --teacher "$run/predictions/teacher.parquet"
          --out "$run/predictions/$study.parquet")
    ;;
  merge)
    args=(--inputs "$run/inputs" --teacher "$run/predictions/teacher.parquet"
          --cellhit "$run/predictions/cellhit.parquet" --parts "$run/predictions" --out "$run/final.parquet")
    ;;
  score)
    books="$root/licob-v0plus"
    args=(--inputs "$run/inputs" --forecast "$run/final.parquet"
          --licob-book "$books/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
          --wong-book "$books/Wong_HCC_PDO_Supplementary_v1.xlsx" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.hcc_3d_study_transfer "$stage" "${args[@]}"
