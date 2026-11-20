# Environment for study R2 (episodic residual adaptation). Sourced by every job of this study.
export R2_ROOT=${R2_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_episodic_residual_r2}
export R2_ASSETS="$R2_ROOT/assets"                 # git-ignored; arrays, checkpoints, run outputs
export R2_PROC="$R2_ASSETS/processed/units_v1"
export R2_RUNS="$R2_ASSETS/runs"
export R2_PUBLIC="$R2_ASSETS/public"               # small tables mirrored into the repository tree by hand
# prior study B: read-only reuse (no copy, no re-download, environment not modified)
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_TSG_SRC="$TB_ROOT/assets/external/TranSiGen"
export TB_PROC="$TB_ROOT/assets/processed"
export TB_RUNS="$TB_ROOT/assets/runs"
export R2_CELLINFO=/mnt/S1/sdata/pubdata/LINCS/LINCS2020/cellinfo_beta.txt
cd "$R2_ROOT"
mkdir -p "$R2_ASSETS/logs" "$R2_PROC" "$R2_RUNS" "$R2_PUBLIC"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R2_COMMIT=$(git rev-parse --short HEAD)
