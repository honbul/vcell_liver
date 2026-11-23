#!/usr/bin/env bash
set -euo pipefail
code=$1
shift
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
vc="$base/vc_build_v1"
root="$base/nextgen_v2/h3_v2"
export PYTHONPATH="$code/src:$root/deps:$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel:$vc/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$root/tmp" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
exec "$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" "$@"
