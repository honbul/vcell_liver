#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --exclude=g1,g2,g3,g4
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=03:00:00

set -euo pipefail
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/followup_20261124/h1_e2"
code=$1
module=$2
shift 2
export PYTHONPATH="$code/p0/src:$code/src:$base/nextgen_v2/h1_v2/deps"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/cache" TORCH_HOME="$root/cache/torch" MPLCONFIGDIR="$root/cache/matplotlib"
exec "$base/nextgen_5track/runtime/bin/python" -m "followup_h1_e2.$module" "$@"
