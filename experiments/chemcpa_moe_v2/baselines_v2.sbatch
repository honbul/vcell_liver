#!/usr/bin/env bash
# The minimum baselines and the v1 small MLPs, re-scored on the v2 truth.
# Usage: sbatch experiments/chemcpa_moe_v2/baselines_v2.sbatch
#SBATCH --job-name=v2_base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/v2_base_%j.log
set -euo pipefail
source experiments/chemcpa_moe_v2/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.baselines \
  --data-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
  --split val --out "$V2_OUT/baselines/val.json"
# the v1 architectures at their v1-selected setting, trained and scored on the v2 truth
for SEED in 42 43 44; do for KIND in A C; do
  "$MOE_PY" -u -m src.moe_drug_v1.train \
    --data-dir "$V2_OUT/data" --manifest-dir manifests/chemcpa_moe_v2 \
    --out-dir "$V2_OUT/runs_mlp/${KIND}_lr3e-3_wd1e-2_s${SEED}" \
    --kind "$KIND" --lr 3e-3 --wd 1e-2 --seed "$SEED" --tag v2
done; done
echo "end=$(date -Is)"
