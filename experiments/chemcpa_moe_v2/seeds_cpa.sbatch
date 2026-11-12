#!/usr/bin/env bash
# The selected setting at the two confirmation seeds.
# Usage: sbatch experiments/chemcpa_moe_v2/seeds_cpa.sbatch <arm> <lr_scale>
#SBATCH --job-name=cpa_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=23:00:00
#SBATCH --output=logs/cpa_seeds_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
ARM=${1:?arm}; S=${2:?lr_scale}
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
for SEED in 43 44; do
  RUN="$V2_OUT/runs/${ARM}_lr${S}_s${SEED}"
  "$CPA_PY" -u -m src.chemcpa_moe_v2.train_cpa \
    --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
    --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
    --hparams configs/chemcpa_moe_v2/hparams_official.json \
    --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
    --out-dir "$RUN" --arm "$ARM" --lr-scale "$S" --seed "$SEED"
done
echo "end=$(date -Is)"
