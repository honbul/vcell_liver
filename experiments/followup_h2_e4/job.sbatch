#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
set -euo pipefail

base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/followup_20261124/h2_e4_loso"
vc="$base/vc_build_v1"
code=$1
module=$2
shift 2
[[ "${SLURM_JOB_NAME:-}" == fu-e4-* && "$code" == "$root/code/"* ]]
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$code/src:$base/followup_20261124/h2_e1e3/deps:$base/nextgen_v2/h2_v2/deps:$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel:$vc/v0plus-model-deps:$vc/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$root/cache" TMPDIR="$root/tmp" MPLCONFIGDIR="$root/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
echo "host=$(hostname) threads=$SLURM_CPUS_PER_TASK GPU=disabled"
exec "$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m "followup_h2_e4.$module" "$@"
