#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail
repository=$1
root=$2
output=$3
storage=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$storage/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
runtime="$storage/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$repository/src:$root/v0plus-model-deps:$root/function/prepare-deps:$runtime"
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
exec "$python" -m vc_build.rna_format_replay \
    --inputs "$root/3d-public/tcga-lihc-rna-format-v1" \
    --package "$root/v0plus-runs/structured-release-6d2c8c6/V0plus" --output "$output"
