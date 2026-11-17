#!/usr/bin/env bash
# R3-0: D1/D2 signal and sample-stability diagnostics on TRAIN drugs (no model).
#SBATCH --job-name=r3_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/r3_diag_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$R3_PY" -u -m src.anchored_distribution_r3.diagnostics
echo "end=$(date -Is)"
