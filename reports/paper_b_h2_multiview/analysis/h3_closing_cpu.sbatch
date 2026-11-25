#!/usr/bin/env bash
#SBATCH --partition=cpu-c3
#SBATCH --nodelist=c3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
root="$base/paper_b_h2_posthoc"
vc="$base/vc_build_v1"
h3="$base/nextgen_v2/h3_v2"
e5="$base/followup_20261124/h2_e5_upgrade/code/d9900bc"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$root/analysis/pins:$root/analysis:$h3/code/full-lincs-v32-capped/src:$e5/src:$h3/deps:$base/followup_20261124/h2_e1e3/deps:$base/nextgen_v2/h2_v2/deps:$vc/v0plus-runs/lincs-pathway-e382c8a-hdf/dependencies/wheel:$vc/v0plus-model-deps:$vc/function/prepare-deps:$base/vcell_liver/vcell_r12_runtime_cu126"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TMPDIR="$root/tmp/h3v2_closing" XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec "$base/nextgen_5track/runtime/bin/python" "$root/analysis/$1" "${@:2}"
