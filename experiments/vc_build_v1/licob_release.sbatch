#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=26G
#SBATCH --time=01:00:00
set -euo pipefail
repository=$1
root=$2
run=$3
stage=$4
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
runs="$root/v0plus-runs"
registered="$runs/licob-cold-5bf53fc"
args=(--inputs "$registered/inputs" --reference "$registered/parts/fold-3.parquet" --results "$registered/external.json")
case "$stage" in
  pack)
    args+=(--priors "$root/v0plus-public-priors" --targets "$root/external-code-v3/drug_targets.json"
           --auxiliary "$runs/depmap-25q3-bb8a47b"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --out "$run/V0plus")
    ;;
  replay)
    export CUDA_VISIBLE_DEVICES=""
    args+=(--package "$run/V0plus" --legacy "$runs/structured-release-6d2c8c6/V0plus"
           --tcga "$runs/rna-format-4ca2dff" --out "$run/replay")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.licob_release "$stage" "${args[@]}"
