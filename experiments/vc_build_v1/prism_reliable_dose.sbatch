#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=00:40:00
set -euo pipefail
repo=$1
root=$2
run=$3
stage=$4
store=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$store/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$store/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repo/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
common=(--run "$run" --historical "$runs/licob-cold-5bf53fc/inputs")
case "$stage" in
  register)
    args=(--core "$runs/source-core-92aa482/core" --extension "$runs/prism-parent-75c35e8/extension"
          --raw "$root/3d-public/prism-20q2-doses-v1")
    ;;
  predict)
    args=(--priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b"
          --case "$5")
    ;;
  score) args=(--supplement "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx") ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.prism_reliable_dose "$stage" "${common[@]}" "${args[@]}"
