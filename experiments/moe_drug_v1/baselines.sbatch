#!/usr/bin/env bash
# Minimum baselines: no-change, per-line mean train Delta, ridge on the same inputs.
# Usage: sbatch experiments/moe_drug_v1/baselines.sbatch [val|test]
#SBATCH --job-name=moe_baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/moe_baselines_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
SPLIT=${1:-val}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) split=$SPLIT start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.baselines \
  --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
  --out-dir assets/moe_drug_v1/baselines --split "$SPLIT"
echo "end=$(date -Is)"
