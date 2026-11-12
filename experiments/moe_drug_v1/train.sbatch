#!/usr/bin/env bash
# One training run. Usage: sbatch experiments/moe_drug_v1/train.sbatch <A|B|C> <lr> <wd> <seed> [tag]
# One GPU per run (plan s3.2): these are small RNA-drug MLPs, not large language models.
#SBATCH --job-name=moe_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/moe_train_%j.log
set -euo pipefail
source experiments/moe_drug_v1/env.sh
KIND=${1:?kind}; LR=${2:?lr}; WD=${3:?wd}; SEED=${4:?seed}; TAG=${5:-}
RUN="assets/moe_drug_v1/runs/${KIND}_lr${LR}_wd${WD}_s${SEED}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
echo "commit=$(git rev-parse HEAD) run=$RUN start=$(date -Is)"
"$MOE_PY" -u -m src.moe_drug_v1.train \
  --data-dir assets/moe_drug_v1/data --manifest-dir manifests/moe_drug_v1 \
  --out-dir "$RUN" --kind "$KIND" --lr "$LR" --wd "$WD" --seed "$SEED" --tag "$TAG"
echo "end=$(date -Is)"
