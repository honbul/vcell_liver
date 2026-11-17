#!/usr/bin/env bash
#SBATCH -J r2_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r2_<name> experiments/episodic_residual_r2/gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/episodic_residual_r2/env.sh
echo "start=$(date -Is) commit=$R2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) args=$*"
"$TB_PY" -m "src.episodic_residual_r2.$1" "${@:2}"
echo "end=$(date -Is)"
