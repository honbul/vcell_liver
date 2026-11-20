# Environment of the R6 v3 track (sensitivity correction; private plan 2026-11-20 section 6).
# Everything this track writes stays under its own worktree (assets/, logs/, reports/); every other path is read-only.
export R6V3_ROOT=${R6V3_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_sensitivity_v3}
export R6V3_ASSETS="$R6V3_ROOT/assets"
# src.r6_specificity_v2 / src.r6_sensitivity_v3 read the treated-well means from $R6F_ASSETS/cache/halves_v1.
# It points at THIS track's assets; halves_v1 is a read-only symlink made by setup_assets.sh (no copy).
export R6F_ASSETS="$R6V3_ASSETS"
export R6V2_ROOT_RO=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_specificity_v2   # READ-ONLY v2
export R6F_V1=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r6_followup_v1          # READ-ONLY follow-up v1
export R6_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_depth_robust_r6        # READ-ONLY original R6
export PRIOR_V2=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2         # READ-ONLY (python env)
export TAHOE_RAW=/mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M                                        # READ-ONLY
export R6_PY="$PRIOR_V2/envs/chemcpa/bin/python"
cd "$R6V3_ROOT"
mkdir -p logs "$R6V3_ASSETS"/{cache,scratch,home}
export HOME_CACHE="$R6V3_ASSETS/home"
export XDG_CACHE_HOME="$HOME_CACHE/xdg" MPLCONFIGDIR="$HOME_CACHE/mpl"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
