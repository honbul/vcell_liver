#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=26G
#SBATCH --time=00:40:00
set -euo pipefail
repo=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repo/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
common=(--historical "$runs/licob-cold-5bf53fc" --run "$run")
case "$stage" in
  predict)
    args=(--extension "$runs/prism-parent-75c35e8/extension" --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b")
    ;;
  score) args=(--supplement "$root/licob-v0plus/NIHMS1969891-supplement-Supplementary_Tables_S1-S13.xlsx") ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.licob_prism_restored "$stage" "${common[@]}" "${args[@]}"
