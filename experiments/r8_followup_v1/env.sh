# Environment of the R8 follow-up (observation transfer v1). Sourced by every r8f_ job.
# The original R8 worktree, its store and its runs are READ-ONLY inputs; every output goes under R8F_ASSETS.
export R8F_ROOT=${R8F_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_followup_v1}
export R8F_ASSETS="$R8F_ROOT/assets"
export R8F_RUNS="$R8F_ASSETS/runs"
export R8F_PUBLIC="$R8F_ASSETS/public"
export R8_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_replicate_calibrated_r8
export R8_PROC="$R8_ORIG/assets/processed/store_v1"          # read-only
export R8_RUNS="$R8_ORIG/assets/runs"                         # read-only (original checkpoints / plate-pair table)
export R8_PUBLIC="$R8F_ASSETS/orig_public_unused"             # the original man_file() falls back to the committed manifests
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_TSG_SRC="$TB_ROOT/assets/external/TranSiGen"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export R2_PROC=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_episodic_residual_r2/assets/processed/units_v1
cd "$R8F_ROOT"
mkdir -p "$R8F_ASSETS/logs" "$R8F_RUNS" "$R8F_PUBLIC" "$R8F_ASSETS/cache"
export HOME_CACHE="$R8F_ASSETS/cache"
export XDG_CACHE_HOME="$HOME_CACHE" TORCH_HOME="$HOME_CACHE/torch" MPLCONFIGDIR="$HOME_CACHE/mpl" NUMBA_CACHE_DIR="$HOME_CACHE/numba"
export JUPYTER_CONFIG_DIR="$HOME_CACHE/jupyter" JUPYTER_DATA_DIR="$HOME_CACHE/jupyter_data" JUPYTER_RUNTIME_DIR="$HOME_CACHE/jupyter_rt" IPYTHONDIR="$HOME_CACHE/ipython"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export R8F_COMMIT=$(git rev-parse --short HEAD)
