#!/usr/bin/env bash
# Generic single-GPU job. Usage: sbatch -J r6f_<name> experiments/r6_followup_v1/run_gpu.sbatch <module> [args...]
# The study never holds more than two GPUs: single jobs are submitted one or two at a time, multi-run
# work goes through pack.sbatch as an array with %2.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/r6_followup_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is) args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
MOD=$1; shift
"$R6_PY" -u -m "src.r6_followup_v1.$MOD" "$@"
echo "end=$(date -Is)"
