#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=00:35:00
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
case "$stage" in
  predict)
    args=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core"
          --extension "$runs/prism-parent-75c35e8/extension" --pdx "$runs/shuen-pdx-6a0e39d"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b")
    ;;
  score) args=(--sources "$root/3d-public/shuen-hcc-pdx-rna-v1/response") ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.vinorelbine_transfer "$stage" --run "$run" "${args[@]}"
