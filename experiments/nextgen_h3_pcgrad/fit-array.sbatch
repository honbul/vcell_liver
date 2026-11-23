#!/usr/bin/env bash
set -euo pipefail
code=$1
exec "$code/experiments/nextgen_h3_pcgrad/run.sbatch" "$code" -m h3_pcgrad.fit "$2" --index "${SLURM_ARRAY_TASK_ID:?}"
