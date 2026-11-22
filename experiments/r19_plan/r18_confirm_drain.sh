#!/usr/bin/env bash
set -euo pipefail

phase=$1
run=$2
if [[ "$phase" == gpu && -n "${SLURM_ARRAY_JOB_ID:-}" && -r "$run/draining-gpu-job" ]]; then
  IFS= read -r draining < "$run/draining-gpu-job"
  if [[ "$SLURM_ARRAY_JOB_ID" == "$draining" ]]; then
    printf 'R18_DEFERRED_BEFORE_FIT %s %s\n' "$draining" "${SLURM_ARRAY_TASK_ID:?}"
    exit 0
  fi
fi
exec bash "$run/r18_confirm.registered.sbatch" "$@"
