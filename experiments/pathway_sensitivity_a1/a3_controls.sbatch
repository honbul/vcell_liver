#!/usr/bin/env bash
#SBATCH --job-name=psa1_a3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-24%8
#SBATCH --output=logs/psa1_a3_%A_%a.log
# Stage A3: permuted-pathway control (tasks 0-19, one per fixed seed) and 50% learning curve (tasks 20-24).
# Submit: sbatch --export=ALL,RUN_ID=<immutable id> experiments/pathway_sensitivity_a1/a3_controls.sbatch
set -euo pipefail
: "${RUN_ID:?set RUN_ID}"
source experiments/pathway_sensitivity_a1/env.sh
i=$SLURM_ARRAY_TASK_ID
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=$RUN_ID task=$i start=$(date -Is)"
if [ "$i" -lt 20 ]; then
  "$PSA1_PY" "$PSA1_SRC/run_internal.py" --run-dir "$PSA1_RUNS/$RUN_ID" --kind A4 --perm-seed $((101 + i))
else
  KINDS=(A0 A1 A1r A2 A3)
  "$PSA1_PY" "$PSA1_SRC/run_internal.py" --run-dir "$PSA1_RUNS/$RUN_ID" --kind "${KINDS[$((i - 20))]}" --fraction 0.5
fi
echo "end=$(date -Is)"
