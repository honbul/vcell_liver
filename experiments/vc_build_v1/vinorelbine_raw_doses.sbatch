#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:30:00
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
common=(--run "$run")
case "$stage" in
  gdsc|prism)
    args=(--original "$root/function/cross-v1" --core "$runs/source-core-92aa482/core"
          --extension "$runs/prism-parent-75c35e8/extension")
    if [[ "$stage" == gdsc ]]; then
      args+=(--raw "$root/v0plus-public-priors/gdsc2-raw-8.5/GDSC2_public_raw_data_27Oct23.zip")
    else
      args+=(--raw "$root/3d-public/prism-20q2-doses-v1")
    fi
    ;;
  predict)
    args=(--core "$runs/source-core-92aa482/core" --pdx "$runs/shuen-pdx-6a0e39d"
          --priors "$root/v0plus-public-priors"
          --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
          --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b")
    ;;
  score) args=(--sources "$root/3d-public/shuen-hcc-pdx-rna-v1/response") ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.vinorelbine_raw_doses "$stage" "${common[@]}" "${args[@]}"
