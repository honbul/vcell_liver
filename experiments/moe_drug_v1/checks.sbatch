#!/usr/bin/env bash
# The four pre-run sanity checks (plan s7.3). Needs a GPU only for the optimisation check.
#SBATCH --job-name=moe_checks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/moe_checks_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.checks \
  --data-dir assets/moe_drug_v1/data \
  --manifest-dir manifests/moe_drug_v1 \
  --prep-dir prior_assets/r3/prep \
  --out reports/moe_drug_v1/checks.json
echo "end=$(date -Is)"
