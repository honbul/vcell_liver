#!/usr/bin/env bash
#SBATCH --job-name=cpa_agg
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpa_agg_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$CPA_PY" -u -m src.chemcpa_moe_v2.aggregate_cpa \
  --runs-dir "${1:-$V2_OUT/runs}" --baselines "$V2_OUT/baselines/val.json/baselines_val.json" \
  --out-dir "${2:-reports/chemcpa_moe_v2}"
echo "end=$(date -Is)"
