#!/usr/bin/env bash
# Generic single-GPU job for R3 python modules.
# Usage: sbatch -J <name> experiments/anchored_distribution_r3/run.sbatch <module> [args...]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is) args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
MOD=$1; shift
"$R3_PY" -u -m "src.anchored_distribution_r3.$MOD" "$@"
echo "end=$(date -Is)"
