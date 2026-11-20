# Environment of the R6 v2 track (specificity / observation, v2; private plan 2026-11-19 section 3).
# Everything this track writes stays under its own worktree's assets/; every other path is read-only.
export R6V2_ROOT=${R6V2_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_specificity_v2}
export R6V2_ASSETS="$R6V2_ROOT/assets"
# src.r6_followup_v1 modules read/write $R6F_ASSETS. It points at THIS track's assets so that any run
# directory is created here; the v1 caches are reached through read-only symlinks made by setup_assets.sh.
export R6F_ASSETS="$R6V2_ASSETS"
export R6F_V1=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_followup_v1       # READ-ONLY follow-up v1
export R6_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_depth_robust_r6     # READ-ONLY original R6
export R6_ORIG_ASSETS="$R6_ORIG/assets"
unset R6_ASSETS
export PRIOR_V2=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2      # READ-ONLY
export R3_CACHE=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_anchored_distribution_r3/assets/cache  # READ-ONLY
export TAHOE_RAW=/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M                                     # READ-ONLY
export R6_PY="$PRIOR_V2/envs/chemcpa/bin/python"
export R6_V2DATA="$PRIOR_V2/v2/data"
cd "$R6V2_ROOT"
mkdir -p logs "$R6V2_ASSETS"/{cache,runs,scratch,home}
export HOME_CACHE="$R6V2_ASSETS/home"
export XDG_CACHE_HOME="$HOME_CACHE/xdg" MPLCONFIGDIR="$HOME_CACHE/mpl" HF_HOME="$HOME_CACHE/hf" TORCH_HOME="$HOME_CACHE/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export WANDB_MODE=disabled
