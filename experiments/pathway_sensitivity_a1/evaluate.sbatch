#!/usr/bin/env bash
#SBATCH --job-name=psa1_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psa1_eval_%j.log
# Score a stage. STAGE=main | controls | external
# Submit: sbatch --export=ALL,STAGE=main,MAIN_RUN=<id>[,CONTROLS_RUN=<id>][,EXT_RUN=<id>] experiments/pathway_sensitivity_a1/evaluate.sbatch
set -euo pipefail
: "${STAGE:?}" "${MAIN_RUN:?}"
source experiments/pathway_sensitivity_a1/env.sh
OUT=reports/pathway_sensitivity_a1/tables
echo "host=$(hostname) commit=$(git rev-parse HEAD) stage=$STAGE start=$(date -Is)"
case "$STAGE" in
  main)     "$PSA1_PY" "$PSA1_SRC/evaluate_internal.py" main --run-dir "$PSA1_RUNS/$MAIN_RUN" --out-dir "$OUT" ;;
  controls) "$PSA1_PY" "$PSA1_SRC/evaluate_internal.py" controls --run-dir "$PSA1_RUNS/$MAIN_RUN" \
              --controls-dir "$PSA1_RUNS/${CONTROLS_RUN:?}" --out-dir "$OUT" ;;
  external) "$PSA1_PY" "$PSA1_SRC/run_external.py" predict --run-dir "$PSA1_RUNS/${EXT_RUN:?}"
            "$PSA1_PY" "$PSA1_SRC/run_external.py" evaluate --run-dir "$PSA1_RUNS/$EXT_RUN" --out-dir "$OUT" ;;
  *) echo "unknown stage"; exit 2 ;;
esac
echo "end=$(date -Is)"
