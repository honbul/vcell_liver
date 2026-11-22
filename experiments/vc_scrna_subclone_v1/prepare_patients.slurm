#!/usr/bin/env bash
#SBATCH --job-name=scrna-v1-patients
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=02:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
public=/mnt/S1/sdata/pubdata
data=$public/Virtual_Cell
longitudinal=$data/GEO/GSE151530/GSE151530
earlier=$data/GEO/GSE125449/GSE125449
output=$root/prepared/scrna-v1/patients
cd "$root/code/vc_scrna_subclone_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [[ ! -f "$output/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_scrna_subclone.prepare_patient \
        --registry "$root/preflight/cohort-identities-v1" \
        --gtf "$public/ngs_ref/gencode/gencode.v47.annotation.gtf.gz" \
        --liver-counts "$data/gssong/vcell_liver/processed/GSE149614.h5ad" \
        --longitudinal "${longitudinal}_matrix.mtx.gz" "${longitudinal}_barcodes.tsv.gz" \
            "${longitudinal}_genes.tsv.gz" "${longitudinal}_Info.txt.gz" \
        --earlier-set1 "${earlier}_Set1_matrix.mtx.gz" "${earlier}_Set1_barcodes.tsv.gz" \
            "${earlier}_Set1_genes.tsv.gz" "${earlier}_Set1_samples.txt.gz" \
        --earlier-set2 "${earlier}_Set2_matrix.mtx.gz" "${earlier}_Set2_barcodes.tsv.gz" \
            "${earlier}_Set2_genes.tsv.gz" "${earlier}_Set2_samples.txt.gz" \
        --output "$output"
fi
