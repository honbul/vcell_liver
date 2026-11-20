#!/usr/bin/env bash
#SBATCH -J r7_final
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# The only job that sets R7_FINAL_EVAL. Store.test_delta still refuses unless the freeze manifest is committed and the tree is clean.
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
export R7_FINAL_EVAL=1
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m src.evidence_gated_r7.final_eval
echo "end=$(date -Is)"
