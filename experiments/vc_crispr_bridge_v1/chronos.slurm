#!/usr/bin/env bash
#SBATCH --job-name=crispr-v1-chronos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=05:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
data=/mnt/S1/sdata/pubdata/DepMap/25Q3
panel=$root/prepared/crispr-v1/chronos-v1
predictions=$root/results/crispr-v1/chronos-predictions
scores=$root/results/crispr-v1/chronos-scores
python=$root/runtime/bin/python
cd "$root/code/vc_crispr_bridge_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if [[ ! -f "$panel/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.chronos_prep --depmap "$data" --output "$panel"
fi
if [[ ! -f "$predictions/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.predict chronos --panel "$panel" --output "$predictions" \
        --store "$root/fm_store_v1/protein-v1" --proteins "$root/prepared/track6-v2/proteins.tsv" --device cuda
fi
if [[ ! -f "$scores/receipt.json" ]]; then
    "$python" -m vc_crispr_bridge.score chronos --panel "$panel" --predictions "$predictions" --output "$scores"
fi
