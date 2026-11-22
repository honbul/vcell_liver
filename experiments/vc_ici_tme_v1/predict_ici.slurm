#!/usr/bin/env bash
#SBATCH --job-name=ici-v1-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_ici_tme_v1"
python=$root/runtime/bin/python
data=$root/preflight/ici-cohorts-v1
output=$root/results/ici-v1
hallmark=/mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/h.all.v2026.1.Hs.symbols.gmt
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for target in GSE202069 GSE140901 GSE141198; do
    if [[ ! -f "$output/predictions/$target/receipt.json" ]]; then
        "$python" -m vc_ici_tme.predict --cohorts "$data" \
            --reference "$root/prepared/ici-v1/cellular-reference" --hallmark "$hallmark" \
            --target "$target" --output "$output/predictions/$target"
    fi
    if [[ ! -f "$output/scores/$target/receipt.json" ]]; then
        "$python" -m vc_ici_tme.score --cohorts "$data" \
            --predictions "$output/predictions/$target" --output "$output/scores/$target"
    fi
done
