# Environment of the R7 XPert v2 track (selection-population correction, OOF calibration, objective x selection).
# Sourced by every r7v2_ job. Everything this track writes goes under $R7V2_ASSETS. The original R7 tree, the R4 units and
# study-B tables are read-only inputs (referenced by absolute path, never copied or modified).
export R7V2_ROOT=${R7V2_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r7_xpert_v2}
export R7V2_ASSETS="$R7V2_ROOT/assets"
export R7V2_RUNS="$R7V2_ASSETS/runs"
export R7V2_PUBLIC="$R7V2_ASSETS/public"          # small publishable outputs, mirrored into the repo with sync_public.sh
# read-only inputs
export R7_ORIG_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_evidence_gated_r7
export R7_ORIG_ASSETS="$R7_ORIG_ROOT/assets"
export XP_UPSTREAM="$R7_ORIG_ASSETS/external/XPert"   # git working copy of GSanShui/XPert (d53ff49), read only
export XP_WORK="$R7_ORIG_ASSETS/xpert_work"           # original R7 XPert inputs (drug features, PPI, h5ad), read only
export XP_PY="$R7_ORIG_ASSETS/envs/xpert/bin/python"  # original R7 XPert env (torch 2.1.0 cu121, SDPA flash_attn shim)
export R4_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_response_geometry_r4
export R4_PROC="$R4_ROOT/assets/processed/units_v1"
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
# the imported original modules resolve paths from these; point every write path into this track's assets
export R7_ASSETS="$R7V2_ASSETS/orig_shadow"         # never written by this track's code (guard: directory not created)
export R7_RUNS="$R7V2_RUNS"
export R7_PUBLIC="$R7V2_PUBLIC"
export R7_EXT="$R7V2_ASSETS/unused_ext"
unset R7_FINAL_EVAL R4_FINAL_EVAL R7V2_FINAL_EVAL
cd "$R7V2_ROOT"
mkdir -p "$R7V2_ASSETS/logs" "$R7V2_RUNS" "$R7V2_PUBLIC" "$R7V2_ASSETS/home" "$R7V2_ASSETS/tmp"
export XDG_CACHE_HOME="$R7V2_ASSETS/home/.cache"
export MPLCONFIGDIR="$R7V2_ASSETS/home/.mpl"
export TORCH_HOME="$R7V2_ASSETS/home/.torch"
export TMPDIR="$R7V2_ASSETS/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R7V2_COMMIT=$(git rev-parse --short HEAD)
export R7_COMMIT=$R7V2_COMMIT
