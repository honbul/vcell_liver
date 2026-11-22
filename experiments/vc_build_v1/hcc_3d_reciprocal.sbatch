#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:15:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
licob="$root/v0plus-runs/licob-author-2f1328b"
wong="$root/v0plus-runs/wong-84d021d"
books="$root/licob-v0plus"
licob_book="$books/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
wong_book="$books/Wong_HCC_PDO_Supplementary_v1.xlsx"
model_args=(
  --priors "$root/v0plus-public-priors"
  --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
  --targets "$root/external-code-v3/drug_targets.json"
)
args=(--inputs "$run/inputs")
case "$stage" in
  prepare)
    args=(--benchmark "$root/v0plus-runs/yang-external-54fcad5/inputs"
          --licob "$licob/cohort" --wong "$wong/cohort" --patient-audit "$licob/patient-audit.json"
          --out "$run/inputs")
    ;;
  teacher|cellhit)
    component=$stage
    if [[ "$stage" == cellhit ]]; then
      shard=$SLURM_ARRAY_TASK_ID
      name="cellhit$shard"
    else
      shard=0
      name=teacher
    fi
    stage=source
    args+=("${model_args[@]}" --component "$component" --shard "$shard"
           --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
           --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
           --upstream "$root/v0plus-upstream" --out "$run/forecasts/$name.parquet")
    ;;
  predict)
    targets=(LICOB Wong)
    fit_books=("$wong_book" "$licob_book")
    study=${targets[$SLURM_ARRAY_TASK_ID]}
    args+=("${model_args[@]}" --study "$study" --fit-book "${fit_books[$SLURM_ARRAY_TASK_ID]}"
           --teacher "$run/forecasts/teacher.parquet" --out "$run/forecasts/$study.parquet")
    ;;
  merge)
    args+=(--parts "$run/forecasts" --out "$run/final.parquet")
    ;;
  score)
    args+=(--forecast "$run/final.parquet" --licob-book "$licob_book" --wong-book "$wong_book"
           --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_reciprocal "$stage" "${args[@]}"
