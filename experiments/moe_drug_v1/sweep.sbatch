#!/usr/bin/env bash
# The identical 6-setting grid for one model, run sequentially inside one GPU allocation.
# Usage: sbatch experiments/moe_drug_v1/sweep.sbatch <A|B|C> [seed]
# Same grid, same seed and same code for every model -- the only thing that differs is --kind.
#SBATCH --job-name=moe_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/moe_sweep_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
KIND=${1:?kind}; SEED=${2:-42}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "commit=$(git rev-parse HEAD) kind=$KIND seed=$SEED start=$(date -Is)"
for LR in 3e-4 1e-3 3e-3; do
  for WD in 1e-4 1e-2; do
    RUN="assets/moe_drug_v1/runs/${KIND}_lr${LR}_wd${WD}_s${SEED}"
    if [ -f "$RUN/record.json" ]; then echo "skip existing $RUN"; continue; fi
    echo "--- $KIND lr=$LR wd=$WD seed=$SEED $(date -Is)"
    "$MOE_PY" -u -m src.moe_drug_v1.train \
      --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
      --out-dir "$RUN" --kind "$KIND" --lr "$LR" --wd "$WD" --seed "$SEED" --tag grid
  done
done
echo "end=$(date -Is)"
