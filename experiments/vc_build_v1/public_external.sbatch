#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=00:45:00
set -euo pipefail

repository=$1
bundle=$2
run=$3
step=$4
targets=$5
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$repository"
outcomes=$bundle/test_labels/outcomes.parquet
case "$step" in
  predict)
    exec "$python" -m vc_build.external --bundle "$bundle" --out "$run" --targets "$targets" ;;
  score)
    score_external() {
      "$python" -m vc_build.evaluate --prediction "$run/predictions.parquet" \
        --observations "$outcomes" --split "$1" --endpoint-direction higher --rank-only \
        --compare direct_structure_rna drug_mean --compare direct_with_targets drug_mean \
        --compare direct_with_targets direct_structure_rna --out "$2"
    }
    score_external X_C_LIMORE "$run/metrics.json"
    if "$python" -c 'import pandas as pd, sys; sys.exit(not pd.read_parquet(sys.argv[1], columns=["track"]).track.eq("X_C_LIMORE_ALL3").any())' "$run/predictions.parquet"; then
      score_external X_C_LIMORE_ALL3 "$run/metrics-all3.json"
    fi ;;
  shift_register)
    exec "$python" -m vc_build.shift_roster --dir "$run/shift" --register-from "$run/predictions.parquet" ;;
  shift_predict)
    "$python" -m vc_build.shift_roster --dir "$run/shift" --select-observations-from "$outcomes"
    exec "$python" -m vc_build.shift --forecast "$run/shift/registered.parquet" \
      --fit "$run/shift/observations_shift_fit.parquet" --interval "$run/shift/observations_interval.parquet" \
      --sentinel "$run/shift/observations_sentinel.parquet" --out "$run/shift/calibrated.parquet" ;;
  shift_score)
    exec "$python" -m vc_build.evaluate --prediction "$run/shift/calibrated.parquet" \
      --observations "$run/shift/observations_score.parquet" --split X_public_shift --endpoint-direction higher \
      --compare direct_structure_rna_source_scale drug_mean_source_scale \
      --compare direct_structure_rna_sentinel_scale direct_structure_rna_source_scale \
      --compare drug_mean_sentinel_scale drug_mean_source_scale --out "$run/shift/metrics.json" ;;
  *) exit 2 ;;
esac
