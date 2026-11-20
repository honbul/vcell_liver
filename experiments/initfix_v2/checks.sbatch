#!/usr/bin/env bash
# The pre-run checks, now including the shared-initialisation identity check that the first build
# did not have. Usage: sbatch experiments/initfix_v2/checks.sbatch
#SBATCH --job-name=ifx_checks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ifx_checks_%j.log
set -euo pipefail
source experiments/initfix_v2/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.checks \
  --data-dir "$MOE_DATA" --manifest-dir manifests/moe_drug_v1 \
  --prep-dir prior_assets/r3/prep \
  --out reports/initfix_v2/checks.json
echo "end=$(date -Is)"
