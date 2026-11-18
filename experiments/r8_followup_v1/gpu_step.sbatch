#!/usr/bin/env bash
#SBATCH -J r8f_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 08:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r8f_<name> experiments/r8_followup_v1/gpu_step.sbatch <module> [args]   (at most one r8f_ GPU job at a time)
set -euo pipefail
source experiments/r8_followup_v1/env.sh
echo "start=$(date -Is) commit=$R8F_COMMIT dirty=$(git status --porcelain -- src configs experiments tests | wc -l) host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) args=$*"
"$TB_PY" -m "$1" "${@:2}"
echo "end=$(date -Is)"
