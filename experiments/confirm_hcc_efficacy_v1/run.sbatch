#!/usr/bin/env bash
set -euo pipefail
workspace=$1
shift
study=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$workspace/src:$study/vc_build_v1/v0plus-model-deps:$study/liver_dili_v1/deps:$study/vc_build_v1/function/prepare-deps:$study/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$study/confirm_20261123/worker1_eff/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
if [[ $1 == --array-fold ]]; then
    shift
    out_parent=$1
    shift
    index=${SLURM_ARRAY_TASK_ID:?}
    exec "$python" "$@" --fold "$index" --out "$out_parent/fold$index"
fi
exec "$python" "$@"
