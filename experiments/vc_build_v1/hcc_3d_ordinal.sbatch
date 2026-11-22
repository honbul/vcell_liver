#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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
pool="$runs/hcc-study-pool-e121f47"
licob="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
wong="$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
args=(--transfer "$runs/hcc-3d-transfer-49a9bee/inputs" --inputs "$pool/inputs")
case "$stage" in
  predict)
    studies=(LICOB Wong YANG2024_HCC)
    holdout=${studies[$SLURM_ARRAY_TASK_ID]}
    args+=(--holdout "$holdout" --archive "$pool/final.parquet" --priors "$root/v0plus-public-priors"
           --targets "$root/external-code-v3/drug_targets.json"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --out "$run/forecasts/$holdout.parquet")
    if [[ "$holdout" != LICOB ]]; then args+=(--licob-book "$licob"); fi
    if [[ "$holdout" != Wong ]]; then args+=(--wong-book "$wong"); fi
    ;;
  score)
    args+=(--licob-book "$licob" --wong-book "$wong" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_ordinal "$stage" "${args[@]}"
