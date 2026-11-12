#!/usr/bin/env bash
#SBATCH --job-name=cpa_exrep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpa_exrep_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
CK=${1:?checkpoint}
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
cd "$CPA_SRC"
echo "host=$(hostname) chemcpa=$(git rev-parse HEAD) start=$(date -Is)"
"$CPA_PY" -u -m src.chemcpa_moe_v2.example_report 2>/dev/null || true
cd "$MOE_ROOT"
"$CPA_PY" -u -m src.chemcpa_moe_v2.example_report \
  --chemcpa-src "$CPA_SRC" --checkpoint "$CK" \
  --dataset-path "$CPA_SRC/project_folder/datasets/sciplex_complete_middle_subset_v2.h5ad" \
  --embedding "$CPA_SRC/project_folder/embeddings/rdkit/sciplex_complete_middle_subset_rdkit2D_embedding.parquet" \
  --out reports/chemcpa_moe_v2/chemcpa_example.json
echo "end=$(date -Is)"
