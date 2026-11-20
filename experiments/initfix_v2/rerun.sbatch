#!/usr/bin/env bash
# The six goal-section-0 re-runs: the already-selected setting (lr 3e-3, wd 1e-2) for A and C at
# seeds 42/43/44, validation only. No new hyper-parameter search, no B, no test.
# Usage: sbatch experiments/initfix_v2/rerun.sbatch
#SBATCH --job-name=ifx_rerun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ifx_rerun_%j.log
set -euo pipefail
source experiments/initfix_v2/env.sh
LR=3e-3; WD=1e-2
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "commit=$(git rev-parse HEAD) start=$(date -Is)"
for SEED in 42 43 44; do
  for KIND in A C; do
    RUN="$MOE_OUT/runs/${KIND}_lr${LR}_wd${WD}_s${SEED}"
    echo "=== $KIND seed=$SEED -> $RUN"
    "$MOE_PY" -u -m src.moe_drug_v1.train \
      --data-dir "$MOE_DATA" --manifest-dir manifests/moe_drug_v1 \
      --out-dir "$RUN" --kind "$KIND" --lr "$LR" --wd "$WD" --seed "$SEED" --tag initfix_v2
  done
done
echo "end=$(date -Is)"
