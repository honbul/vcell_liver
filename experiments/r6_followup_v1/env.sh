# Environment of the R6 follow-up (specificity / observation, v1).
# Everything this study writes stays under its own worktree's assets/; every other path is read-only.
export R6F_ROOT=${R6F_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_followup_v1}
export R6F_ASSETS="$R6F_ROOT/assets"                     # git-ignored: caches, runs, checkpoints, predictions
export R6_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_depth_robust_r6   # READ-ONLY original R6
export R6_ORIG_ASSETS="$R6_ORIG/assets"                  # READ-ONLY (count cache, contract, 24 final checkpoints)
# R6_ASSETS is deliberately NOT exported: an original R6 entry point started from here by mistake
# fails on the missing variable instead of writing into the original run directory.
unset R6_ASSETS
export PRIOR_V2=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2   # READ-ONLY
export R3_CACHE=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_anchored_distribution_r3/assets/cache  # READ-ONLY (wells.tsv only)
export TAHOE_RAW=/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M  # READ-ONLY public mirror (parquet shards)
export R6_PY="$PRIOR_V2/envs/chemcpa/bin/python"         # reused read-only (torch 2.8, pyarrow, sklearn)
export R6_V2DATA="$PRIOR_V2/v2/data"
cd "$R6F_ROOT"
mkdir -p logs "$R6F_ASSETS"/{cache,runs,scratch,home}
export HOME_CACHE="$R6F_ASSETS/home"                     # compute nodes may have no home directory
export XDG_CACHE_HOME="$HOME_CACHE/xdg" MPLCONFIGDIR="$HOME_CACHE/mpl" HF_HOME="$HOME_CACHE/hf" TORCH_HOME="$HOME_CACHE/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export WANDB_MODE=disabled
