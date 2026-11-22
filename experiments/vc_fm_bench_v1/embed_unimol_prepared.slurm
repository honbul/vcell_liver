#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:40:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1
"$root/runtime/bin/python" -m vc_fm_bench.embed --model unimol --device cuda --batch-size 32 \
    --input "$root/prepared/track6-v2/molecules.tsv" --checkpoint "$root/checkpoints/unimol" \
    --unimol-conformers "$root/prepared/track6-unimol-coordinates-v2" \
    --output "$root/fm_store_v1/unimol-bounded-v2"
