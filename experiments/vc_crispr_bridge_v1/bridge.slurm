#!/usr/bin/env bash
#SBATCH --job-name=crispr-v1-bridge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=06:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
data=/mnt/S1/sdata/pubdata/DepMap/25Q3
source=${root%/nextgen_5track}/vc_build_v1/function/limore-v1
work=$root/prepared/crispr-v1
results=$root/results/crispr-v1
python=$root/runtime/bin/python
cd "$root/code/vc_crispr_bridge_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [[ ! -f "$work/bridge-panels-native-v1/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.bridge_panels --panels "$root/prepared/track6-panels-v2" \
        --targets "$root/preflight/bridge-targets-native-v1" --output "$work/bridge-panels-native-v1"
fi
if [[ ! -f "$results/bridge-predictions/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.predict bridge --panel "$work/bridge-panels-native-v1" \
        --output "$results/bridge-predictions" --chemistry "$root/fm_store_v1/chemistry-v1" \
        --context "$root/prepared/limore-rna-v1/rna.parquet" --hgnc "$data/Gene.csv" \
        --chronos "$data/CRISPRGeneEffect.csv" --depmap-models "$data/Model.csv" \
        --cells "$source/cell_registry.tsv" --hepg2 "$work/HepG2" \
        --hepg2-predictions "$results/hepg2-predictions" --gene-panels "$work/gene-panels-v1" \
        --device cuda
fi
if [[ ! -f "$results/bridge-scores/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.score bridge --panel "$work/bridge-panels-native-v1" \
        --predictions "$results/bridge-predictions" --output "$results/bridge-scores"
fi
