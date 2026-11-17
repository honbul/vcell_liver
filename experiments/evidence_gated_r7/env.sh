# Environment for study R7 (evidence-gated mechanism conditioning). Sourced by every job of this study.
export R7_ROOT=${R7_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_evidence_gated_r7}
export R7_ASSETS="$R7_ROOT/assets"                 # git-ignored: downloads, KG cache, runs, checkpoints, predictions
export R7_EXT="$R7_ASSETS/external"                # ChEMBL / OmniPath / XPert downloads (pinned, hashed)
export R7_KG="$R7_ASSETS/kg_v1"                    # built drug-target evidence + graph cache
export R7_RUNS="$R7_ASSETS/runs"
export R7_PUBLIC="$R7_ASSETS/public"               # small publishable outputs, mirrored into the repo with sync_public.sh
# R4 units (CMap 2020 Level 3, 10 uM / 24 h, 978 landmarks, R4 drug-group split): read-only reuse, never written
export R4_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_response_geometry_r4
export R4_PROC="$R4_ROOT/assets/processed/units_v1"
# prior study B: read-only (ECFP4 array, drug table, gene axis, python env)
export TB_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1
export TB_PY="$TB_ROOT/assets/envs/tsg/bin/python"
export TB_PROC="$TB_ROOT/assets/processed/cmap2020_v1"
export XP_PY="$R7_ASSETS/envs/xpert/bin/python"    # separate env for the XPert comparator
export HF_HUB_DISABLE_TELEMETRY=1                  # UniMol weights (public, no account) without telemetry
cd "$R7_ROOT"
mkdir -p "$R7_ASSETS/logs" "$R7_EXT" "$R7_KG" "$R7_RUNS" "$R7_PUBLIC"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export R7_COMMIT=$(git rev-parse --short HEAD)
# generic client identity for every external request (no user / project / path information)
export R7_UA="Mozilla/5.0 (X11; Linux x86_64) research-client"
unset R4_FINAL_EVAL
