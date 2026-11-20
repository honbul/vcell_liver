# Environment of R8 unseen-drug selection v3 (E2-B outer/inner drug-split correction). Sourced by every v3r8_ job.
# The original R8 store, the R8 v2 / follow-up v1 worktrees and their assets are READ-ONLY inputs; every output goes under R8V3_ASSETS.
export R8V3_ROOT=${R8V3_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r8_unseen_drug_v3}
export R8V3_ASSETS="$R8V3_ROOT/assets"
export R8V3_RUNS="$R8V3_ASSETS/runs"
export R8_ORIG=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_replicate_calibrated_r8
export R8_PROC="$R8_ORIG/assets/processed/store_v1"                       # read-only
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"                       # same interpreter as the v2 sbatch files
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"                   # read-only (ecfp4.npy, drugs.tsv)
export R2_PROC=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_episodic_residual_r2/assets/processed/units_v1   # read-only
# writers of earlier tracks must never run here
export R8V2_RUNS=/nonexistent_r8v2_runs R8V2_PUBLIC=/nonexistent_r8v2_public R8F_RUNS=/nonexistent_r8f_runs R8F_PUBLIC=/nonexistent_r8f_public
cd "$R8V3_ROOT"
mkdir -p "$R8V3_ASSETS/logs" "$R8V3_RUNS" "$R8V3_ASSETS/cache"
export XDG_CACHE_HOME="$R8V3_ASSETS/cache" TORCH_HOME="$R8V3_ASSETS/cache/torch" MPLCONFIGDIR="$R8V3_ASSETS/cache/mpl"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export R8V3_COMMIT=$(git rev-parse --short HEAD)
