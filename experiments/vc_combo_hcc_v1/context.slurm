#!/usr/bin/env bash
#SBATCH --job-name=combo-v1-context
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=5
#SBATCH --mem=35G
#SBATCH --time=01:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
pub=/mnt/S1/sdata/pubdata/DepMap/25Q3
raw=$root/raw/combo
work=$root/prepared/combo-v1
python=$root/runtime/bin/python
cd "$root/code/vc_combo_hcc_v1"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1

if [[ ! -f "$work/context/receipt.json" ]]; then
    "$python" -m vc_combo_hcc.context --assays "$work/assays" \
        --rna "$pub/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv" \
        --chronos "$pub/CRISPRGeneEffect.csv" --genes "$pub/Gene.csv" --models "$pub/Model.csv" \
        --sanger-structures "$root/preflight/gdsc-combo-structures-v1" \
        --bridge-targets "$root/preflight/bridge-targets-v1" --drugcomb-metadata "$raw/DrugComb_drug_identifiers.xlsx" \
        --sanger-matrices "$raw/"*_matrix_results.csv --gdsc-monotherapy "$raw/GDSC2_fitted_dose_response_25Feb20.csv" \
        --output "$work/context"
fi
