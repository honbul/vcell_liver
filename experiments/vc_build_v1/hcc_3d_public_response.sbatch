#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
runs="$root/v0plus-runs"
pool="$runs/hcc-study-pool-e121f47"
licob="$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx"
wong="$root/licob-v0plus/Wong_HCC_PDO_Supplementary_v1.xlsx"
args=(--transfer "$runs/hcc-3d-transfer-49a9bee/inputs" --inputs "$pool/inputs"
      --public "$root/3d-public/hcc-3d-context-v1")
case "$stage" in
  predict)
    cohorts=(LICOB Wong)
    holdout=${cohorts[$SLURM_ARRAY_TASK_ID]}
    args+=(--holdout "$holdout" --archive "$pool/final.parquet" --out "$run/forecasts/$holdout.parquet")
    if [[ "$holdout" == LICOB ]]; then
      args+=(--wong-book "$wong")
    else
      args+=(--licob-book "$licob")
    fi
    ;;
  score)
    args+=(--licob-book "$licob" --wong-book "$wong" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_public_response "$stage" "${args[@]}"
