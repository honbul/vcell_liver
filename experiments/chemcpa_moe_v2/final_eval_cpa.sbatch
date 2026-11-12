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
# the minimum baselines on the test split, and the v1 architectures at their already-fixed
# setting, so every number in the final table comes from this one pass
"$MOE_PY" -u -m src.moe_drug_v1.baselines \
  --data-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --split test --out-dir "$V2_OUT/baselines/test"
"$MOE_PY" -u -m src.moe_drug_v1.final_eval \
  --data-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --runs-dir "$V2_OUT/runs_mlp" \
  --selection configs/chemcpa_moe_v2/selection_mlp.json \
  --baselines-dir "$V2_OUT/baselines/test" \
  --out-dir reports/chemcpa_moe_v2/mlp_test

"$CPA_PY" -u -m src.chemcpa_moe_v2.final_eval_cpa \
  --baselines-test "$V2_OUT/baselines/test/baselines_test.json" \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
  --hparams configs/chemcpa_moe_v2/hparams_official.json \
  --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --runs-dir "$V2_OUT/runs_e30" --selection reports/chemcpa_moe_v2/selection.json \
  --out-dir reports/chemcpa_moe_v2 \
  --freeze-manifest manifests/chemcpa_moe_v2/freeze.json \
  --pred-dir "$V2_OUT/test_preds"
echo "end=$(date -Is)"
