#!/usr/bin/env bash
# Generic single-GPU job for R6 python modules.
# Usage: sbatch -J r6_<name> experiments/depth_robust_r6/run_gpu.sbatch <module> [args...]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/depth_robust_r6/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is) args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
MOD=$1; shift
"$R6_PY" -u -m "src.depth_robust_r6.$MOD" "$@"
echo "end=$(date -Is)"
