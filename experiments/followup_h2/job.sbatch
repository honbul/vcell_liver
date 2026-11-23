#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/followup_20261124/h2_e1e3"
vc="$base/vc_build_v1"
code=$1
shift
export PYTHONPATH="$code/src:$root/deps:$base/nextgen_v2/h2_v2/deps:$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel:$vc/v0plus-model-deps:$vc/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/deps/cache" TMPDIR="$root/deps/tmp" MPLCONFIGDIR="$root/deps/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec "$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m "followup_h2.$1" "${@:2}"
