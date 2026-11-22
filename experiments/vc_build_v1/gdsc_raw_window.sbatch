#!/usr/bin/env bash
#SBATCH --cpus-per-task=3
#SBATCH --mem=18G
#SBATCH --time=01:00:00
set -euo pipefail

repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
public=/mnt/S1/sdata/pubdata/DepMap/DrugScreens
export PYTHONPATH="$repository/src:$root/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
bundle="$root/function/cross-v1"

case "$stage" in
  extract)
    args=(--bundle "$bundle"
          --raw "$root/v0plus-public-priors/gdsc2-raw-8.5/GDSC2_public_raw_data_27Oct23.zip"
          --out "$run/gdsc")
    ;;
  score)
    args=(--bundle "$bundle" --gdsc "$run/gdsc"
          --ctrp-post-qc "$public/CTRP_CTD2/v20.data.per_cpd_post_qc.txt"
          --treatments "$root/v0plus-runs/dose-window-3e0e6dc/prism20q2-treatment.csv"
          --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.gdsc_raw_window "$stage" "${args[@]}"
