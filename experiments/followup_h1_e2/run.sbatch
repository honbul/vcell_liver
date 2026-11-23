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
runtime="$base/nextgen_5track/runtime/bin/python"
export PYTHONPATH="$code/src:$base/nextgen_v2/h1_v2/deps"
if [[ $module == p0 ]]; then
    runtime="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
    export PYTHONPATH="$code/p0/src:$code/src:$base/vc_build_v1/v0plus-model-deps:$base/liver_dili_v1/deps:$base/vc_build_v1/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/cache" TORCH_HOME="$root/cache/torch" MPLCONFIGDIR="$root/cache/matplotlib"
exec "$runtime" -m "followup_h1_e2.$module" "$@"
