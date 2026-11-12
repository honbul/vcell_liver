#!/usr/bin/env bash
# The learning-rate scan for one arm: the official rates times 0.3, 1 and 3, seed 42, validation
# only. Usage: sbatch experiments/chemcpa_moe_v2/sweep_cpa.sbatch <P0|P1|P2|P3>
#SBATCH --job-name=cpa_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=logs/cpa_sweep_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
ARM=${1:?arm}
export PYTHONPATH="$CPA_SRC:${PYTHONPATH:-}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "commit=$(git rev-parse HEAD) chemcpa=$(cd $CPA_SRC && git rev-parse HEAD) start=$(date -Is)"
for S in 0.3 1.0 3.0; do
  RUN="$V2_OUT/runs/${ARM}_lr${S}_s42"
  echo "=== $ARM lr_scale=$S -> $RUN"
  "$CPA_PY" -u -m src.chemcpa_moe_v2.train_cpa \
    --chemcpa-src "$CPA_SRC" --adata "$V2_OUT/tahoe_cells.h5ad" \
    --embedding "$V2_OUT/tahoe_rdkit2D_embedding.parquet" \
    --hparams configs/chemcpa_moe_v2/hparams_official.json \
    --v2-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
    --out-dir "$RUN" --arm "$ARM" --lr-scale "$S" --seed 42
done
echo "end=$(date -Is)"
