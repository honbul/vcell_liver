#!/usr/bin/env bash
#SBATCH --job-name=ici-v1-reference
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_ici_tme_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
output=$root/prepared/ici-v1/cellular-reference
if [[ ! -f "$output/receipt.json" ]]; then
    "$root/runtime/bin/python" -m vc_ici_tme.reference \
        --patients "$root/prepared/scrna-v1/patients" \
        --liana "$root/raw/ici/liana_omni.csv" --output "$output"
fi
