#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export HF_HOME="$root/runtime-cache/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 USE_TF=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4

"$root/runtime/bin/python" -m vc_fm_bench.embed --model esm2 \
    --input "$root/prepared/track6-v2/proteins.tsv" --checkpoint "$root/checkpoints/esm2" \
    --output "$root/fm_store_v1/protein-v1/esm2"
