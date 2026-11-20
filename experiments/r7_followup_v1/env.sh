# Environment of the R7 follow-up (target attribution v1). Sourced by every r7f_ job.
# Everything this study writes goes under $R7F_ASSETS; the original R7 / R4 / study-B trees are read-only inputs.
export R7F_ROOT=${R7F_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r7_followup_v1}
export R7F_ASSETS="$R7F_ROOT/assets"
export R7F_RUNS="$R7F_ASSETS/runs"
export R7F_CACHE="$R7F_ASSETS/cache"               # label-free caches (null mappings, reachability)
export R7F_PUBLIC="$R7F_ASSETS/public"             # small publishable outputs, mirrored into the repo with sync_public.sh
# read-only inputs
export R7_ORIG_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_evidence_gated_r7
export R7_ORIG_RUNS="$R7_ORIG_ROOT/assets/runs"    # original checkpoints (read only: D2 sensitivity, seed-matched backbones of the final re-evaluation)
export R7_KG="$R7_ORIG_ROOT/assets/kg_v1"          # original knowledge cache (read only; kg_build / rewire are never run here)
export R4_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_response_geometry_r4
export R4_PROC="$R4_ROOT/assets/processed/units_v1"
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
# the imported original modules resolve their write paths from these: point them into this study's assets
export R7_RUNS="$R7F_RUNS"
export R7_PUBLIC="$R7F_PUBLIC"
export R7_EXT="$R7F_ASSETS/unused_ext"
unset R7_FINAL_EVAL R4_FINAL_EVAL
cd "$R7F_ROOT"
mkdir -p "$R7F_ASSETS/logs" "$R7F_RUNS" "$R7F_CACHE" "$R7F_PUBLIC" "$R7F_ASSETS/home" "$R7F_ASSETS/tmp"
# nodes may lack a home directory: every cache goes into this study's assets
export HOME_FALLBACK="$R7F_ASSETS/home"
export XDG_CACHE_HOME="$R7F_ASSETS/home/.cache"
export MPLCONFIGDIR="$R7F_ASSETS/home/.mpl"
export TORCH_HOME="$R7F_ASSETS/home/.torch"
export TMPDIR="$R7F_ASSETS/tmp"
export OMP_NUM_THREADS=${R7F_THREADS:-${SLURM_CPUS_PER_TASK:-4}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R7_COMMIT=$(git rev-parse --short HEAD)
export R7F_COMMIT=$R7_COMMIT
