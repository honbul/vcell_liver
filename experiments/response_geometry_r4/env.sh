# Environment for study R4 (response geometry). Sourced by every job of this study.
export R4_ROOT=${R4_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_response_geometry_r4}
export R4_ASSETS="$R4_ROOT/assets"                 # git-ignored: arrays, checkpoints, predictions, run outputs
export R4_PROC="$R4_ASSETS/processed/units_v1"
export R4_RUNS="$R4_ASSETS/runs"
export R4_PUBLIC="$R4_ASSETS/public"               # small publishable outputs, mirrored into the repo with sync_public.sh
export R4_WAVE="$R4_ASSETS/external/WAVE"          # upstream WAVE, pinned e27e453b
# prior study B: read-only reuse (no copy, no re-download, environment not modified)
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export R4_CELLINFO=/mnt/S1/sdata/pubdata/LINCS/LINCS2020/cellinfo_beta.txt
cd "$R4_ROOT"
mkdir -p "$R4_ASSETS/logs" "$R4_PROC" "$R4_RUNS" "$R4_PUBLIC"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R4_COMMIT=$(git rev-parse --short HEAD)
