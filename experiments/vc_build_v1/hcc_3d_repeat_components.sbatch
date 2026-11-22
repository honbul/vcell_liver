#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2
licob=(--licob "$root/v0plus-runs/licob-wide-rna-2833a20/cohort"
       --licob-labels "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx")
case "$stage" in
  repeats)
    args=(--public "$root/3d-public/hcc-3d-context-v1" --out "$run/repeats.json")
    ;;
  predict)
    args=("${licob[@]}" --forecast "$root/v0plus-runs/source-gated-8b2be38/forecasts/LICOB.parquet"
          --reference "$repository/reports/vc_build_v1/v0plus/source_gated_verification.json"
          --out "$run/LICOB_pairs.parquet")
    ;;
  score)
    args=("${licob[@]}" --forecast "$run/LICOB_pairs.parquet" --out "$run/licob_pairs_score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.hcc_3d_repeat_components "$stage" "${args[@]}"
