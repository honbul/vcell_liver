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
prior="$root/v0plus-runs/hcc-3d-reciprocal-116897d"
books="$root/licob-v0plus"
book_args=(--licob-book "$books/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
           --wong-book "$books/Wong_HCC_PDO_Supplementary_v1.xlsx")
args=(--inputs "$prior/inputs" --prior "$prior/final.parquet")
case "$stage" in
  prepare)
    args+=("${book_args[@]}" --out "$run/splits")
    ;;
  predict)
    studies=(LICOB Wong)
    study=${studies[$((SLURM_ARRAY_TASK_ID / 5))]}
    repeat=$((SLURM_ARRAY_TASK_ID % 5))
    args+=(--splits "$run/splits" --study "$study" --repeat "$repeat"
           --priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json"
           --out "$run/forecasts/$study-$repeat.parquet")
    ;;
  score)
    args+=("${book_args[@]}" --splits "$run/splits" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_fewshot "$stage" "${args[@]}"
