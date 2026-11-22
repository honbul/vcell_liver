#!/usr/bin/env bash
#SBATCH --job-name=fm-v1-unimol-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=01:30:00
set -euo pipefail

root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_5track
cd "$root/code/vc_fm_bench_v1"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$root/runtime/bin/python" -m vc_fm_bench.conformers --workers "$SLURM_CPUS_PER_TASK" \
    --input "$root/prepared/track6-v2/molecules.tsv" --output "$root/prepared/track6-unimol-coordinates-v2"
