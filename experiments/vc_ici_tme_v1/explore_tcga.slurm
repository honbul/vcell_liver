#!/usr/bin/env bash
#SBATCH --job-name=ici-v1-tcga
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:30:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_ici_tme_v1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
data=$root/raw/ici/tcga
output=$root/results/ici-v1/tcga-exploratory
if [[ ! -f "$output/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_ici_tme.tcga \
        --expression /mnt/S1/sdata/pubdata/GDCdata/TCGA-LIHC/harmonized_STAR \
        --listing "$data/nextgen_lihc_star_files.json" --subtypes "$data/nextgen-thorsson.tsv.gz" \
        --reference "$root/prepared/ici-v1/cellular-reference" \
        --hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/h.all.v2026.1.Hs.symbols.gmt \
        --output "$output"
fi
