#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=assets/logs/%x_%A_%a.out
# Generic runner: sbatch -J r10_<name> [resources] scripts/r10/py.sbatch <module> <args...>
# In an array job the token @LINE@ in any argument becomes A549 / K562 / MCF7 (task 0 / 1 / 2).
set -euo pipefail
source scripts/r10/env.sh
LINES=(A549 K562 MCF7)
ARGS=()
for a in "$@"; do
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then a=${a//@LINE@/${LINES[$SLURM_ARRAY_TASK_ID]}}; fi
  ARGS+=("$a")
done
echo "cmd: ${ARGS[*]}"
/usr/bin/time -v $PY -P -m "${ARGS[@]}"
echo "end=$(date -Is)"
