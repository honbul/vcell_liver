#!/usr/bin/env bash
#SBATCH --job-name=fm-v1-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
"$root/runtime/bin/python" -m vc_fm_bench.report --results "$root/results/track6-v1" \
    --store "$root/fm_store_v1" --output "$root/results/track6-v1/report-v1"
