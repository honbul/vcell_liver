#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
"$root/runtime/bin/python" -m vc_fm_bench.embed --model clamp --device cuda --batch-size 64 \
    --input "$root/prepared/track6-v2/molecules.tsv" --checkpoint "$root/checkpoints/clamp" \
    --output "$root/fm_store_v1/chemistry-v1/clamp"
