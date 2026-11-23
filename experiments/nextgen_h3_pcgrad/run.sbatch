#!/usr/bin/env bash
set -euo pipefail
code=$1
shift
gssong=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$gssong/nextgen_h123/h3_pcgrad"
export PYTHONPATH="$code/src:$gssong/vc_build_v1/v0plus-model-deps:$gssong/liver_dili_v1/deps:$gssong/vc_build_v1/function/prepare-deps:$gssong/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$root/tmp" CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export H3_CONTRACT="$code/experiments/nextgen_h3_pcgrad/contract.json"
exec "$gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" "$@"
