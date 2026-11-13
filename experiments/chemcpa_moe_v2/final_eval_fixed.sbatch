#!/usr/bin/env bash
# The test pass for the re-run after the covariate fix (2026-11-13).
#
# This is a bug-fix re-evaluation, not a new blind test: the test drug groups were scored
# twice already with the broken evaluator. What makes it admissible is the order: the runs, the
# selection and the checkpoint hashes are committed before this job runs, and nothing it
# produces is used to change any of them. Only the chemCPA arms are scored; the baselines' and
# the v1 MLPs' test predictions did not go through the broken path, are reused from the earlier
# pass, and are scored here on the arms' rows with the arms' functions.
#SBATCH --job-name=cpa_final_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpa_final_fix_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
test -z "$(git status --porcelain -- src experiments configs manifests reports/chemcpa_moe_v2/fixed)" \
  || { echo "refusing to open the test split with uncommitted code or selection"; exit 1; }
test -f manifests/chemcpa_moe_v2/freeze_fixed_pretest.json \
  || { echo "the pre-test checkpoint hashes are not committed"; exit 1; }
"$CPA_PY" -u -m src.chemcpa_moe_v2.final_eval_cpa \
  --baselines-test "$V2_OUT/baselines/test/baselines_test.json" \
  --baseline-preds-dir "$V2_OUT/baselines/test" \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
  --hparams configs/chemcpa_moe_v2/hparams_official.json \
  --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --runs-dir "$V2_OUT/runs_fix_e30" --selection reports/chemcpa_moe_v2/fixed/e30/selection.json \
  --out-dir reports/chemcpa_moe_v2/fixed \
  --freeze-manifest manifests/chemcpa_moe_v2/freeze_fixed.json \
  --pred-dir "$V2_OUT/test_preds_fix"
echo "end=$(date -Is)"
