#!/usr/bin/env bash
# v2 condition table, chemCPA AnnData and drug embedding, from the v2 raw pass.
#SBATCH --job-name=v2_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2_prep_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$CPA_PY" -u -m src.chemcpa_moe_v2.prepare_v2 \
  --raw-dir "$V2_OUT/raw" --prep-dir prior_assets/r3/prep \
  --v1-manifest-dir manifests/moe_drug_v1 \
  --out-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2
"$CPA_PY" -u -m src.chemcpa_moe_v2.make_anndata \
  --raw-dir "$V2_OUT/raw" --prep-dir prior_assets/r3/prep \
  --v1-manifest-dir manifests/moe_drug_v1 \
  --out "$V2_OUT/tahoe_cells.h5ad" \
  --manifest-out manifests/chemcpa_moe_v2/anndata.json
"$CPA_PY" -u -m src.chemcpa_moe_v2.make_embedding \
  --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
  --out "$V2_OUT/tahoe_rdkit2D_embedding.parquet"
echo "end=$(date -Is)"
