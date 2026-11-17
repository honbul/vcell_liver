# Environment for study R5 (noise-aware perturbation design). Sourced by every job of this study.
export R5_ROOT=${R5_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_noise_aware_design_r5}
export R5_ASSETS="$R5_ROOT/assets"                 # git-ignored: arrays, run outputs
export R5_PROC="$R5_ASSETS/processed/store_v1"
export R5_RUNS="$R5_ASSETS/runs"
export R5_PUBLIC="$R5_ASSETS/public"               # small tables mirrored into the repository by sync_public.sh
# prior study B: read-only reuse (no copy, no re-download, environment not modified)
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export TB_RAW="$TB_ROOT/assets/raw"
export LINCS_META=/mnt/S1/sdata/pubdata/LINCS/LINCS2020
cd "$R5_ROOT"
mkdir -p "$R5_ASSETS/logs" "$R5_PROC" "$R5_RUNS" "$R5_PUBLIC"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R5_COMMIT=$(git rev-parse --short HEAD)
