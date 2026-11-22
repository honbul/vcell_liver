#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:40:00
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
runs="$root/v0plus-runs"
transfer="$runs/hcc-3d-transfer-49a9bee/inputs"
reciprocal="$runs/hcc-3d-reciprocal-116897d"
wide="$runs/licob-wide-rna-2833a20"
licob="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
wong="$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
case "$stage" in
  prepare)
    args=(--transfer "$transfer" --reciprocal "$reciprocal/inputs" --reciprocal-forecast "$reciprocal/final.parquet"
          --wide "$wide/cohort" --wide-forecast "$wide/final.parquet"
          --yang-forecast "$runs/yang-external-54fcad5/final.parquet" --out "$run/inputs")
    ;;
  predict)
    studies=(LICOB Wong YANG2024_HCC)
    holdout=${studies[$SLURM_ARRAY_TASK_ID]}
    args=(--transfer "$transfer" --inputs "$run/inputs" --holdout "$holdout"
          --priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --out "$run/parts/$holdout.parquet")
    if [[ "$holdout" != LICOB ]]; then args+=(--licob-book "$licob"); fi
    if [[ "$holdout" != Wong ]]; then args+=(--wong-book "$wong"); fi
    ;;
  merge)
    args=(--inputs "$run/inputs" --parts "$run/parts" --out "$run/final.parquet")
    ;;
  score)
    args=(--inputs "$run/inputs" --transfer "$transfer" --licob-book "$licob" --wong-book "$wong"
          --forecast "$run/final.parquet" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_study_pool "$stage" "${args[@]}"
