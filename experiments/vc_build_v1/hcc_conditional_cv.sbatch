#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
set -euo pipefail

repo=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
study_python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo/src:$root/v0plus-model-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
bundle="$root/function/cross-v1"
case "$stage" in
  predict)
    assays=(pooled CTRPv2 GDSC2 PRISM20Q2)
    assay=${assays[$((SLURM_ARRAY_TASK_ID / 3))]}
    fold=$((SLURM_ARRAY_TASK_ID % 3))
    args=(
      --bundle "$bundle" --priors "$root/v0plus-public-priors"
      --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
      --targets "$root/external-code-v3/drug_targets.json"
      --auxiliary "$root/v0plus-runs/depmap-25q3-bb8a47b"
      --baseline "$root/v0plus-runs/structured-bf73a0a" --assay "$assay" --fold "$fold"
      --out "$run/forecasts/$assay-$fold.parquet"
    )
    ;;
  score)
    args=(--bundle "$bundle" --forecasts "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$study_python" -m vc_build.hcc_conditional_cv "$stage" "${args[@]}"
