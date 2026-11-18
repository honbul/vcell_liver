#!/usr/bin/env bash
#SBATCH -J r7_addendum
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 04:00:00
#SBATCH -o assets/logs/%x_%j.out
# Addendum A-2 one-time evaluation (needs the committed addendum freeze; Store.test_delta keeps its own checks).
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
export R7_FINAL_EVAL=1
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m src.evidence_gated_r7.addendum_eval
echo "end=$(date -Is)"
