#!/usr/bin/env bash
# The comparison's preconditions, before any arm is scored.
#SBATCH --job-name=cpa_checks
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpa_checks_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$CPA_PY" -u -m src.chemcpa_moe_v2.checks_cpa \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
  --hparams configs/chemcpa_moe_v2/hparams_official.json \
  --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --out reports/chemcpa_moe_v2/checks.json
echo "end=$(date -Is)"
