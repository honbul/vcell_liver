#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
benchmark="$root/v0plus-runs/yang-external-54fcad5"
args=(--inputs "$benchmark/inputs")
case "$stage" in
  linear|molecular)
    component=$stage
    stage=fit
    args+=(--component "$component" --reference "$benchmark/final.parquet"
           --settings "$root/v0plus-runs/final-8e5d3f4/settings.json"
           --priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json"
           --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
           --out "$run/components/$component")
    ;;
  pack)
    args+=(--score "$benchmark/score.json" --parts "$run/components" --out "$run/V0plus")
    ;;
  replay)
    export CUDA_VISIBLE_DEVICES=""
    args+=(--package "$run/V0plus" --reference "$benchmark/final.parquet" --out "$run/replay")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.v0plus_corrected_release "$stage" "${args[@]}"
