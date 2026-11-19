# Environment of R8 observation-transfer v2 (corrections P0 + E1/E2 follow-up). Sourced by every r8v2_ job.
# The original R8 worktree, the R8 follow-up v1 worktree and their assets are READ-ONLY inputs; every output goes under R8V2_ASSETS.
export R8V2_ROOT=${R8V2_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_transfer_v2}
export R8V2_ASSETS="$R8V2_ROOT/assets"
export R8V2_RUNS="$R8V2_ASSETS/runs"
export R8V2_PUBLIC="$R8V2_ASSETS/public"
export R8F_V1_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_followup_v1      # read-only
export R8F_V1_RUNS="$R8F_V1_ROOT/assets/runs"                                                   # read-only
export R8F_RUNS=/nonexistent_r8f_v1_runs_not_writable_from_v2                                   # v1 writers must never run here
export R8F_PUBLIC=/nonexistent_r8f_v1_public_not_writable_from_v2
export R8_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_replicate_calibrated_r8
export R8_PROC="$R8_ORIG/assets/processed/store_v1"          # read-only
export R8_RUNS="$R8_ORIG/assets/runs"                         # read-only
export R8_PUBLIC="$R8V2_ASSETS/orig_public_unused"
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export R2_PROC=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_episodic_residual_r2/assets/processed/units_v1
cd "$R8V2_ROOT"
mkdir -p "$R8V2_ASSETS/logs" "$R8V2_RUNS" "$R8V2_PUBLIC" "$R8V2_ASSETS/cache"
export HOME_CACHE="$R8V2_ASSETS/cache"
export XDG_CACHE_HOME="$HOME_CACHE" TORCH_HOME="$HOME_CACHE/torch" MPLCONFIGDIR="$HOME_CACHE/mpl" NUMBA_CACHE_DIR="$HOME_CACHE/numba"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export R8V2_COMMIT=$(git rev-parse --short HEAD)
