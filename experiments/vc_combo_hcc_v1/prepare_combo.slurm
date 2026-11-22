#!/usr/bin/env bash
#SBATCH --job-name=combo-v1-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=02:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
raw=$root/raw/combo
work=$root/prepared/combo-v1
python=$root/runtime/bin/python
cd "$root/code/vc_combo_hcc_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

if [[ ! -f "$work/sanger-zip/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.zip_wells --wells "$raw/matrix_well_stats.zip" \
        --output "$work/sanger-zip"
fi
if [[ ! -f "$work/assays/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.prepare_assays --drugcomb "$raw/DrugComb_v1.5.csv" \
        --depmap /mnt/S1/sdata/pubdata/DepMap/25Q3/Model.csv --drugs "$raw/DrugComb_drug_identifiers.xlsx" \
        --cells "$raw/DrugComb_cell_line_identifiers.xlsx" --sanger "$raw/"*_matrix_results.csv \
        --sanger-structures "$root/preflight/gdsc-combo-structures-v1" --sanger-zip "$work/sanger-zip" \
        --output "$work/assays"
fi
if [[ ! -f "$work/panels/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.splits --assays "$work/assays/assays.parquet" --output "$work/panels"
fi
