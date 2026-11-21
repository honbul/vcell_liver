#!/usr/bin/env bash
set -euo pipefail

stage="$1"
namespace="$2"
lab="$(dirname "$(dirname "$namespace")")"
snapshot="$3"
commit="$4"
cd "$snapshot"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS" MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
r11="$lab/vcell_liver_r11_chemical_probe/assets"
r14="$lab/vcell_liver_r14_cross_organ"
options=(
  --cache "$r14/execution-v1"
  --index "$r14/metadata/v4"
  --t3 "$lab/vcell_liver_goal_tracks/v1/t3"
  --genes "$r11/metadata/genes.tsv"
  --lincs "$namespace/lincs"
  --work "$namespace/heparg"
  --protocol "$snapshot/reports/p123/PROTOCOL.md"
  --code-commit "$commit"
)
if [[ "$stage" == predict ]]; then
    datasets=(GSE296254 GSE284321)
    task="$SLURM_ARRAY_TASK_ID"
    options+=(--dataset "${datasets[task / 3]}" --fold "$((task % 3))")
fi
printf 'job=%s task=%s host=%s start=%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-none}" "$(hostname)" "$(date -Is)"
exec /usr/bin/time -v "$r11/study-env/bin/python" -u -m src.p123.p3_run "$stage" "${options[@]}"
