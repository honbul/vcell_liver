#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=15G
#SBATCH --time=00:50:00
set -euo pipefail
code=$1
root=$2
run=$3
action=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
runs="$root/v0plus-runs"
pool="$runs/hcc-study-pool-e121f47"
licob="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
wong="$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
tki="$runs/tki-gr50-6361497"
args=(--inputs "$pool/inputs" --tki "$tki")
case "$action" in
  predict)
    studies=(LICOB Wong YANG2024_HCC GR50)
    holdout=${studies[${SLURM_ARRAY_TASK_ID:?}]}
    args+=(--holdout "$holdout" --old "$pool/final.parquet" --transfer "$runs/hcc-3d-transfer-49a9bee/inputs"
           --expanded-wong "$runs/wong-expanded-15b1345/inputs" --out "$run/parts/$holdout.parquet")
    if [[ "$holdout" != LICOB ]]; then args+=(--licob-book "$licob"); fi
    if [[ "$holdout" != Wong ]]; then args+=(--wong-book "$wong"); fi
    ;;
  merge)
    args+=(--parts "$run/parts" --out "$run/final.parquet")
    ;;
  score)
    args+=(--forecast "$run/final.parquet" --transfer "$runs/hcc-3d-transfer-49a9bee/inputs"
           --licob-book "$licob" --wong-book "$wong" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_anchors "$action" "${args[@]}"
