#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/nextgen_v2/h2_v2"
vc="$base/vc_build_v1"
code=$1
shift
export PYTHONPATH="$code/src:$root/deps:$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel:$vc/v0plus-model-deps:$vc/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/deps/cache"
export TMPDIR="$root/deps/tmp"
export MPLCONFIGDIR="$root/deps/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec "$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m "nextgen_h2_v2.$1" "${@:2}"
