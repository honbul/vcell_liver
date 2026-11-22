#!/usr/bin/env bash
#SBATCH --job-name=scrna-v1-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
work=$root/prepared/scrna-v1
result=$root/results/scrna-v1
python=$root/runtime/bin/python
cd "$root/code/vc_scrna_subclone_v1"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export NUMBA_CACHE_DIR="$root/runtime-cache/numba" MPLCONFIGDIR="$root/runtime-cache/matplotlib"

if [[ ! -f "$result/transfer-scores/receipt.json" ]]; then
    "$python" -m vc_scrna_subclone.score --predictions "$result/transfer-predictions" \
        --sources "$work/source-panels-v1" --singlecell "$work/singlecell-v1" \
        --longitudinal "$work/nci-timepoints-v1/nci_timepoints.tsv" --output "$result/transfer-scores"
fi
if [[ ! -f "$result/figures/receipt.json" ]]; then
    "$python" -m vc_scrna_subclone.figures --predictions "$result/transfer-predictions" \
        --scores "$result/transfer-scores" --singlecell "$work/singlecell-v1" \
        --scgpt "$root/fm_store_v1/scrna-v1/scgpt" --output "$result/figures"
fi
