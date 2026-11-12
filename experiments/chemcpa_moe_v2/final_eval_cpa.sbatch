#!/usr/bin/env bash
# The single test pass. Everything must be committed before this runs.
#SBATCH --job-name=cpa_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cpa_final_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
test -z "$(git status --porcelain -- src experiments configs manifests)" \
  || { echo "refusing to open the test split with uncommitted code"; exit 1; }
"$CPA_PY" -u -m src.chemcpa_moe_v2.final_eval_cpa \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
  --hparams configs/chemcpa_moe_v2/hparams_official.json \
  --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --runs-dir "$V2_OUT/runs" --selection reports/chemcpa_moe_v2/selection.json \
  --out-dir reports/chemcpa_moe_v2 \
  --freeze-manifest manifests/chemcpa_moe_v2/freeze.json
echo "end=$(date -Is)"
