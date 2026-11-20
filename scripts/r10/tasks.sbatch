#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=assets/logs/%x_%A_%a.out
# Task-table runner (v2): sbatch -J r10v2_<name> --array=0-N%C scripts/r10/tasks.sbatch <tasks.txt>
# Line i (0-based) of tasks.txt holds the python module and its arguments for array task i.
set -euo pipefail
source scripts/r10/env.sh
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
echo "cmd: $LINE"
/usr/bin/time -v $PY -P -m $LINE
echo "end=$(date -Is)"
