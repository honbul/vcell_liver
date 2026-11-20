#!/usr/bin/env bash
# Freeze and open the test split once, for every model with the same code.
# Run only after selection.json exists and the models are final.
#SBATCH --job-name=moe_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/moe_final_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.baselines \
  --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
  --out-dir assets/moe_drug_v1/baselines --split test
"$MOE_PY" -u -m src.moe_drug_v1.final_eval \
  --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
  --runs-dir assets/moe_drug_v1/runs --selection reports/moe_drug_v1/selection.json \
  --baselines-dir assets/moe_drug_v1/baselines --out-dir reports/moe_drug_v1
echo "end=$(date -Is)"
