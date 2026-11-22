#!/usr/bin/env bash
#SBATCH --job-name=fm-v1-bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
source=${root%/nextgen_5track}/vc_build_v1/function
code=$root/code/vc_fm_bench_v1
panels=$root/prepared/track6-panels-v2
store=$root/fm_store_v1
output=$root/results/track6-v1
python=$root/runtime/bin/python
cd "$code"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

if (( $# == 0 )); then
    printf 'Provide at least one panel ID\n' >&2
    exit 2
fi

# Fit/freeze every supplied panel before beginning scoring. Each prediction
# process opens its training and query files and never receives a label path.
for name in "$@"; do
    [[ $name =~ ^[a-z0-9-]+$ && -f "$panels/$name/receipt.json" ]] || exit 2
    [[ ! -f "$output/predictions/$name/receipt.json" ]] || continue
    options=()
    if [[ $name =~ ^(gdsc2|limore)- ]]; then
        options+=(--weights "$root/raw/pathways/signed_weights.tsv")
        options+=(--hallmark /mnt/S1/sdata/pubdata/Virtual_Cell/MSigDB/h.all.v2026.1.Hs.symbols.gmt)
        if [[ $name == limore-drug-cold-* ]]; then
            options+=(--cell "$store/limore-v1" --rna "$root/prepared/limore-rna-v1/rna.parquet")
        else
            options+=(--cell "$store/bulk-v1" --rna "$source/cross-v1/train/rna.parquet" \
                "$source/cross-v1/test_inputs/rna.parquet")
        fi
    fi
    "$python" -m vc_fm_bench.benchmark predict --panel "$panels/$name" \
        --chem "$store/chemistry-v1" --output "$output/predictions/$name" "${options[@]}"
done

for name in "$@"; do
    [[ ! -f "$output/scores/$name/receipt.json" ]] || continue
    options=()
    if [[ $name == limore-cell-cold ]]; then
        options=(--external "$source/cross-v1/test_labels/outcomes.parquet" \
            --prepared "$root/prepared/track6-v2")
    fi
    "$python" -m vc_fm_bench.benchmark score --panel "$panels/$name" \
        --predictions "$output/predictions/$name" --output "$output/scores/$name" "${options[@]}"
done
