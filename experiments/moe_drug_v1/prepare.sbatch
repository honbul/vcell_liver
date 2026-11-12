#!/usr/bin/env bash
# Build the frozen (cell line, drug) table from the reused R3 prep assets. CPU only -- no GPU is held
# for preprocessing (plan s3.2). Usage: sbatch experiments/moe_drug_v1/prepare.sbatch [extra args]
#SBATCH --job-name=moe_prepare
#SBATCH --partition=cpu
#SBATCH --exclude=g1,g2,g3,g4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/moe_prepare_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
/usr/bin/time -v "$MOE_PY" -u src/moe_drug_v1/prepare.py \
  --prep-dir prior_assets/r3/prep \
  --out-dir assets/moe_drug_v1/data \
  --manifest-dir manifests/moe_drug_v1 "$@"
echo "end=$(date -Is)"
