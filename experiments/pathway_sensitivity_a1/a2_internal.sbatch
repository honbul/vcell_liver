#!/usr/bin/env bash
#SBATCH --job-name=psa1_a2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/psa1_a2_%A_%a.log
# Stage A2: A0, A1, A1r, A2, A3 under the same nested family-group evaluation.
# Submit: sbatch --export=ALL,RUN_ID=<immutable id> experiments/pathway_sensitivity_a1/a2_internal.sbatch
set -euo pipefail
: "${RUN_ID:?set RUN_ID}"
source experiments/pathway_sensitivity_a1/env.sh
KINDS=(A0 A1 A1r A2 A3)
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=$RUN_ID kind=${KINDS[$SLURM_ARRAY_TASK_ID]} start=$(date -Is)"
"$PSA1_PY" "$PSA1_SRC/run_internal.py" --run-dir "$PSA1_RUNS/$RUN_ID" --kind "${KINDS[$SLURM_ARRAY_TASK_ID]}"
echo "end=$(date -Is)"
