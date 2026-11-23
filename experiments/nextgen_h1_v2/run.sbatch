#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --time=06:00:00
set -euo pipefail
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/nextgen_v2/h1_v2"
code=$1
shift
export PYTHONPATH="$code/src:$root/deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/cache" TORCH_HOME="$root/cache/torch" MPLCONFIGDIR="$root/cache/matplotlib"
exec "$base/nextgen_5track/runtime/bin/python" -m "nextgen_h1_v2.$1" "${@:2}"
