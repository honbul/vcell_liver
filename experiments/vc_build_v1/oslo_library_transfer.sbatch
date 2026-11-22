#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
set -euo pipefail
code=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$code/src:$root/v0plus-model-deps:$root/function/prepare-deps:$storage/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
prepared="$root/v0plus-runs/oslo-public-5187b3f"
args=(--inputs "$prepared/registered" --bridge "$prepared/dose_bridge.json" --run "$run")
case "$stage" in
  fit)
    args+=(--priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt)
    ;;
  score) ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.oslo_library_transfer "$stage" "${args[@]}"
