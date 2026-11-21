#!/usr/bin/env bash
set -euo pipefail

stage="$1"
cache="$2"
index="$3"
work="$4"
snapshot="$5"
code_commit="$6"
python="$7"
cd "$snapshot"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
options=(--cache "$cache" --index "$index" --work "$work" --protocol "$snapshot/reports/goal_tracks/PROTOCOL.md" --code-commit "$code_commit")
if [[ "$stage" == predict ]]; then
  datasets=(GSE296254 GSE284321)
  task="$SLURM_ARRAY_TASK_ID"
  options+=(--dataset "${datasets[task / 3]}" --fold "$((task % 3))")
fi
exec /usr/bin/time -v "$python" -u -m src.goal_tracks.heparg "$stage" "${options[@]}"
