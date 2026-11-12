#!/usr/bin/env bash
# Collect the run records, select one setting per model on validation, apply the promotion gate.
#SBATCH --job-name=moe_aggregate
#SBATCH --partition=cpu
#SBATCH --exclude=g1,g2,g3,g4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/moe_aggregate_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.aggregate \
  --runs-dir assets/moe_drug_v1/runs \
  --baselines assets/moe_drug_v1/baselines/baselines_val.json \
  --out-dir reports/moe_drug_v1
echo "end=$(date -Is)"
