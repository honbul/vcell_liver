# Environment for study R3 (mean-anchored residual diffusion on real Tahoe single cells).
# Everything this study writes stays under its own worktree; the prior study's assets are read-only.
export R3_ROOT=${R3_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_anchored_distribution_r3}
export R3_ASSETS="$R3_ROOT/assets"                      # git-ignored: caches, checkpoints, generated cells
export PRIOR_V2=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2   # READ-ONLY
export R3_PY="$PRIOR_V2/envs/chemcpa/bin/python"        # reused as is (torch 2.8, anndata, sklearn); never modified
export R3_H5AD="$PRIOR_V2/v2/tahoe_cells.h5ad"
export R3_V2DATA="$PRIOR_V2/v2/data"
cd "$R3_ROOT"
mkdir -p logs "$R3_ASSETS"/{cache,runs,gen,scratch}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export WANDB_MODE=disabled
