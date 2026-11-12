#!/usr/bin/env bash
# One pass over the Tahoe shards for the v2 truth and the fixed single-cell sample.
# Usage: sbatch experiments/chemcpa_moe_v2/build_v2.sbatch [cells_per_group]
#SBATCH --job-name=v2_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=28
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v2_build_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
K=${1:-64}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.chemcpa_moe_v2.build_v2 \
  --prep-dir prior_assets/r3/prep \
  --v1-manifest-dir manifests/moe_drug_v1 \
  --v1-data-dir "$V1_DATA" \
  --out-dir "$V2_OUT/raw" \
  --cells-per-group "$K" --procs 14 --parts 1
echo "end=$(date -Is)"
