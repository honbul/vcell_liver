#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
set -euo pipefail
root=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
work="$root/nextgen_h123/h1_anchor"
code=$1
shift
export PYTHONPATH="$code/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
exec "$root/nextgen_5track/runtime/bin/python" -m "nextgen_h1_anchor.$1" "${@:2}"
