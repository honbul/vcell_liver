#!/usr/bin/env bash
#SBATCH --job-name=fm-v1-string
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
"$root/runtime/bin/python" -m vc_fm_bench.string_graph \
    --links /mnt/S1/sdata/pubdata/Virtual_Cell/STRING_v12/9606.protein.links.detailed.v12.0.txt.gz \
    --aliases "$root/raw/string/9606.protein.aliases.v12.0.txt.gz" \
    --output "$root/fm_store_v1/protein-v1/string"
