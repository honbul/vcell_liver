#!/usr/bin/env bash
#SBATCH --job-name=scrna-v1-embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=08:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
work=$root/prepared/scrna-v1
output=$root/fm_store_v1/scrna-v1/scgpt
results=$root/results/scrna-v1/transfer-predictions
cd "$root/code/vc_scrna_subclone_v1"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$root/runtime-cache/hf"
export PYTHONPATH="$root/code/vc_fm_bench_v1${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

if [[ ! -f "$output/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_scrna_subclone.singlecell_embeddings \
        --singlecell "$work/singlecell-v1" --sources "$work/source-panels-v1" \
        --checkpoint /mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/checkpoints/scgpt-whole-human \
        --output "$output" --device cuda
fi
if [[ ! -f "$work/organoid-anchors-v1/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_scrna_subclone.organoid_anchors \
        --sources "$work/source-panels-v1" --output "$work/organoid-anchors-v1"
fi
if [[ ! -f "$results/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_scrna_subclone.predict \
        --singlecell "$work/singlecell-v1" --sources "$work/source-panels-v1" \
        --anchors "$work/organoid-anchors-v1" --scgpt "$output" \
        --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/h.all.v2026.1.Hs.symbols.gmt \
        --output "$results" --device cuda
fi
