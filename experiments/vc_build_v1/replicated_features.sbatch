#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=01:15:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
base=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core")
case "$stage" in
  predict)
    assays=(pooled CTRPv2 GDSC2 PRISM20Q2)
    assay=${assays[$((SLURM_ARRAY_TASK_ID / 3))]}
    fold=$((SLURM_ARRAY_TASK_ID % 3))
    part=(--assay "$assay" --fold "$fold" --baseline "$runs/expanded-panel-81119c8"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json"
          --out "$run/forecasts/$assay-$fold.parquet")
    ;;
  score)
    part=(--forecast "$run/forecasts" --out "$run/score.json")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.replicated_features "$stage" "${base[@]}" "${part[@]}"
