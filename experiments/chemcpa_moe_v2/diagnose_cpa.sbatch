#!/usr/bin/env bash
# Post-selection diagnostics on the frozen checkpoints: validation only, nothing is selected here.
#SBATCH --job-name=cpa_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cpa_diag_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
RUNS=${1:-$V2_OUT/runs}
OUT=${2:-reports/chemcpa_moe_v2/diagnostics.json}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) runs=$RUNS start=$(date -Is)"
"$CPA_PY" -u -m src.chemcpa_moe_v2.diagnose_cpa \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
  --hparams configs/chemcpa_moe_v2/hparams_official.json \
  --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --runs-dir "$RUNS" --selection "${3:-reports/chemcpa_moe_v2/selection.json}" \
  --out "$OUT"
echo "end=$(date -Is)"
