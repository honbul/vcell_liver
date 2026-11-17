#!/usr/bin/env bash
#SBATCH -J r8_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 08:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r8_<name> experiments/replicate_calibrated_r8/gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/replicate_calibrated_r8/env.sh
echo "start=$(date -Is) commit=$R8_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) args=$*"
"$TB_PY" -m "src.replicate_calibrated_r8.$1" "${@:2}"
echo "end=$(date -Is)"
