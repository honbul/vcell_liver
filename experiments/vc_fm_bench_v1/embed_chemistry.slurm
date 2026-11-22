#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
code=$root/code/vc_fm_bench_v1
store=$root/fm_store_v1/chemistry-v1
cd "$code"
export HF_HOME="$root/runtime-cache/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4

status=0
for model in ecfp4 chemberta2 molformer unimol; do
    if [[ -f "$store/$model/receipt.json" ]]; then
        continue
    fi
    device=cuda
    checkpoint=(--checkpoint "$root/checkpoints/$model")
    if [[ $model == molformer ]]; then
        checkpoint=(--checkpoint "$root/checkpoints/molformer-transformers4")
    fi
    if [[ $model == ecfp4 ]]; then
        device=cpu
        checkpoint=()
    fi
    echo "Embedding $model"
    if ! "$root/runtime/bin/python" -m vc_fm_bench.embed --model "$model" \
        --input "$root/prepared/track6-v2/molecules.tsv" "${checkpoint[@]}" \
        --device "$device" --output "$store/$model"; then
        echo "$model failed" >&2
        status=1
    fi
done
exit "$status"
