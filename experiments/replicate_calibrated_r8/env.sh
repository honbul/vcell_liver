# Environment for study R8 (replicate-calibrated, state-preserving adaptation). Sourced by every job of this study.
export R8_ROOT=${R8_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_replicate_calibrated_r8}
export R8_ASSETS="$R8_ROOT/assets"                 # git-ignored: index store, run outputs, checkpoints, caches
export R8_PROC="$R8_ASSETS/processed/store_v1"
export R8_RUNS="$R8_ASSETS/runs"
export R8_PUBLIC="$R8_ASSETS/public"               # small tables mirrored into the repository by sync_public.sh
# earlier studies: read-only reuse (no copy, no re-download, environments not modified)
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_TSG_SRC="$TB_ROOT/assets/external/TranSiGen"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export TB_RAW="$TB_ROOT/assets/raw/cmap2020"
export R2_PROC=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_episodic_residual_r2/assets/processed/units_v1
export R5_PROC=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_noise_aware_design_r5/assets/processed/store_v1
cd "$R8_ROOT"
mkdir -p "$R8_ASSETS/logs" "$R8_PROC" "$R8_RUNS" "$R8_PUBLIC" "$R8_ASSETS/cache"
# nodes may lack a home directory: every cache goes under the study assets
export HOME_CACHE="$R8_ASSETS/cache"
export XDG_CACHE_HOME="$HOME_CACHE" TORCH_HOME="$HOME_CACHE/torch" MPLCONFIGDIR="$HOME_CACHE/mpl" NUMBA_CACHE_DIR="$HOME_CACHE/numba"
export JUPYTER_CONFIG_DIR="$HOME_CACHE/jupyter" JUPYTER_DATA_DIR="$HOME_CACHE/jupyter_data" JUPYTER_RUNTIME_DIR="$HOME_CACHE/jupyter_rt" IPYTHONDIR="$HOME_CACHE/ipython"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export R8_COMMIT=$(git rev-parse --short HEAD)
