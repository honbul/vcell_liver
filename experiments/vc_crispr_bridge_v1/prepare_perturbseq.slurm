#!/usr/bin/env bash
#SBATCH --job-name=crispr-v1-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=10:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
data=/mnt/S1/sdata/pubdata/Virtual_Cell
code=$root/code/vc_crispr_bridge_v1
work=$root/prepared/crispr-v1
cd "$code"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=2
python=$root/runtime/bin/python

prepare() {
    local assay=$1 file=$2
    [[ ! -f "$work/$assay/receipt.json" ]] || return 0
    "$python" -m vc_crispr_bridge.perturbseq --source "$file" --assay "$assay" --output "$work/$assay"
}

prepare K562-essential "$data/Replogle2022/K562_essential_raw_singlecell_01.h5ad"
prepare RPE1 "$data/Replogle2022/rpe1_raw_singlecell_01.h5ad"
prepare Jurkat "$data/GEO/GSE264667/GSE264667_jurkat_raw_singlecell_01.h5ad"
prepare HepG2 "$data/GEO/GSE264667/GSE264667_hepg2_raw_singlecell_01.h5ad"

if [[ ! -f "$work/gene-panels-v1/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.design --source "$work/K562-essential" "$work/RPE1" "$work/Jurkat" \
        --hepg2 "$work/HepG2" --output "$work/gene-panels-v1"
fi
