#!/usr/bin/env bash
# Descriptive statistics for the data card (structure similarity, truth magnitude, plate blocks).
#SBATCH --job-name=moe_describe
#SBATCH --partition=cpu
#SBATCH --exclude=g1,g2,g3,g4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/moe_describe_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.describe \
  --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
  --out reports/moe_drug_v1/data_description.json
echo "end=$(date -Is)"
