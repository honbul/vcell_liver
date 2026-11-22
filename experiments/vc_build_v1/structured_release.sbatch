#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
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
comparison="$runs/limore-corrected-4a986af"
args=(--core "$runs/source-core-92aa482/core")
case "$stage" in
  pack)
    args+=(--settings "$runs/final-8e5d3f4/settings.json" --score "$comparison/score.json"
           --priors "$root/v0plus-public-priors"
           --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt
           --targets "$root/external-code-v3/drug_targets.json" --auxiliary "$runs/depmap-25q3-bb8a47b"
           --out "$run/V0plus")
    ;;
  replay)
    export CUDA_VISIBLE_DEVICES=""
    args+=(--package "$run/V0plus" --original "$root/function/cross-v1" --reference "$comparison/final.parquet"
           --wong "$runs/wong-tissue-a4120a9/cli-replay" --licob "$runs/licob-cold-5bf53fc/inputs"
           --out "$run/replay")
    ;;
  *) exit 2 ;;
esac
exec "$python" -m vc_build.structured_release "$stage" "${args[@]}"
