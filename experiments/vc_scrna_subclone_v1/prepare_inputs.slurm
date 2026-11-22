#!/usr/bin/env bash
#SBATCH --job-name=scrna-v1-inputs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
src=${root%/nextgen_5track}/vc_build_v1/function
work=$root/prepared/scrna-v1
python=$root/runtime/bin/python
cd "$root/code/vc_scrna_subclone_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [[ ! -f "$work/source-panels-v1/receipt.json" ]]; then
    "$python" -m vc_scrna_subclone.source_panels --prepared "$root/prepared/track6-v2" \
        --rna "$src/cross-v1/train/rna.parquet" "$root/prepared/limore-rna-v1/rna.parquet" \
        --drugs "$src/limore-v1/compound_registry.tsv" \
        --organoid-counts "$root/raw/GSE182593/counts.txt.gz" \
        --organoid-series "$root/raw/GSE182593/series_matrix.txt.gz" \
        --output "$work/source-panels-v1"
fi
if [[ ! -f "$work/singlecell-v1/receipt.json" ]]; then
    "$python" -m vc_scrna_subclone.singlecell_inputs --patients "$work/patients" \
        --sources "$work/source-panels-v1" --cell-registry "$src/limore-v1/cell_registry.tsv" \
        --tahoe /mnt/S1/sdata/pubdata/Virtual_Cell/Tahoe-100M/by_cell_line/SNU-423_counts.h5ad \
        --output "$work/singlecell-v1"
fi
