#!/usr/bin/env bash
#SBATCH --job-name=r1_r0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r1_r0_%j.log
# Stage R1-0: Track A predictors on the fixed-ID gene axis + scorer check.
# Submit: sbatch --export=ALL,RUN_ID=<immutable id> experiments/softprior_multitask_r1/r0_track_a.sbatch
set -euo pipefail
: "${RUN_ID:?}"
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=$RUN_ID start=$(date -Is)"
"$R1_PY" "$R1_SRC/run_track_a_models.py" --run-dir "$R1_RUNS/$RUN_ID"
"$R1_PY" "$R1_SRC/eval_r0.py" --run-dir "$R1_RUNS/$RUN_ID" --out-dir reports/softprior_multitask_r1/tables
echo "end=$(date -Is)"
