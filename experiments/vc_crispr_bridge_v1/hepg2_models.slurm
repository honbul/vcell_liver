#!/usr/bin/env bash
#SBATCH --job-name=crispr-v1-hepg2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=06:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
panel=$root/prepared/crispr-v1/gene-panels-v1
predictions=$root/results/crispr-v1/hepg2-predictions
scores=$root/results/crispr-v1/hepg2-scores
data=/mnt/S1/sdata/pubdata
python=$root/runtime/bin/python
cd "$root/code/vc_crispr_bridge_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [[ ! -f "$predictions/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.predict hepg2 --panel "$panel" --output "$predictions" \
        --store "$root/fm_store_v1/protein-v1" --proteins "$root/prepared/track6-v2/proteins.tsv" \
        --hgnc "$data/DepMap/25Q3/Gene.csv" \
        --links "$data/Virtual_Cell/STRING_v12/9606.protein.links.detailed.v12.0.txt.gz" \
        --aliases "$root/raw/string/9606.protein.aliases.v12.0.txt.gz" --device cuda
fi

if [[ ! -f "$scores/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.score hepg2 --panel "$panel" --predictions "$predictions" --output "$scores"
fi
