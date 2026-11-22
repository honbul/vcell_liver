#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
public=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/checkpoints
source=${root%/nextgen_5track}/vc_build_v1/function
code=$root/code/vc_fm_bench_v1
store=$root/fm_store_v1/bulk-v1
inputs=("$source/cross-v1/train/rna.parquet" "$source/cross-v1/test_inputs/rna.parquet")
if [[ ${SLURM_JOB_NAME:-} == fm-v1-limore ]]; then
    store=$root/fm_store_v1/limore-v1
    inputs=("$root/prepared/limore-rna-v1/rna.parquet")
fi
cd "$code"
export HF_HOME="$root/runtime-cache/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4

status=0
for model in scgpt geneformer; do
    if [[ -f "$store/$model/receipt.json" ]]; then
        continue
    fi
    opts=(--checkpoint "$public/scgpt-whole-human")
    if [[ $model == geneformer ]]; then
        opts=(--checkpoint "$public/Geneformer-V2-104M" --assets "$root/checkpoints/geneformer_assets")
    fi
    echo "Embedding $model bulk features"
    if ! "$root/runtime/bin/python" -m vc_fm_bench.transcriptome --model "$model" \
        --input "${inputs[@]}" \
        --scale log2 "${opts[@]}" --output "$store/$model"; then
        echo "$model failed" >&2
        status=1
    fi
done
exit "$status"
