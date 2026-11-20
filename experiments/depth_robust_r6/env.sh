# Environment for study R6 (depth-robust response representation).
# Everything this study writes stays under its own worktree's assets/; every other path is read-only.
export R6_ROOT=${R6_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_depth_robust_r6}
export R6_ASSETS="$R6_ROOT/assets"                       # git-ignored: caches, runs, checkpoints, predictions
export PRIOR_V2=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2   # READ-ONLY
export R3_CACHE=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_anchored_distribution_r3/assets/cache  # READ-ONLY
export TAHOE_RAW=/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M  # READ-ONLY public mirror (parquet shards)
export R6_PY="$PRIOR_V2/envs/chemcpa/bin/python"         # reused as is (torch 2.8, pyarrow, sklearn); never modified
export R6_V2DATA="$PRIOR_V2/v2/data"
cd "$R6_ROOT"
mkdir -p logs "$R6_ASSETS"/{cache,runs,scratch,home}
# compute nodes may have no home directory
export HOME_CACHE="$R6_ASSETS/home"
export XDG_CACHE_HOME="$HOME_CACHE/xdg" MPLCONFIGDIR="$HOME_CACHE/mpl" HF_HOME="$HOME_CACHE/hf" TORCH_HOME="$HOME_CACHE/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export WANDB_MODE=disabled
