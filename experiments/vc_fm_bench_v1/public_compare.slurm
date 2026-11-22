#!/usr/bin/env bash
#SBATCH --job-name=fm-v1-public
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
source=${root%/nextgen_5track}/vc_build_v1/function
prepared=$root/prepared/track6-v2
panels=$root/prepared/track6-panels-v2
store=$root/fm_store_v1
output=$root/results/track6-v1
python=$root/runtime/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3

for name in limore-cell-cold dili-release-increment dili-scaffold-{0..4}; do
    fit=()
    scoring=()
    if [[ $name == limore-cell-cold ]]; then
        fit=(--cell "$store/bulk-v2" --rna "$source/cross-v1/train/rna.parquet"
             "$source/cross-v1/test_inputs/rna.parquet")
        scoring=(--prepared "$prepared" --external "$source/cross-v1/test_labels/outcomes.parquet")
    fi
    if [[ ! -f "$output/public_predictions/$name/receipt.json" ]]; then
        "$python" -m vc_fm_bench.public_models --panel "$panels/$name" --prepared "$prepared" \
            --chem "$store/chemistry-v1" --output "$output/public_predictions/$name" "${fit[@]}"
    fi
    "$python" -m vc_fm_bench.benchmark score --panel "$panels/$name" \
        --predictions "$output/public_predictions/$name" --output "$output/public_scores/$name" "${scoring[@]}"
done
