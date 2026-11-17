#!/usr/bin/env bash
#SBATCH -J r4_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r4_<name> experiments/response_geometry_r4/gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/response_geometry_r4/env.sh
echo "start=$(date -Is) commit=$R4_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
"$TB_PY" -m "src.response_geometry_r4.$1" "${@:2}"
echo "end=$(date -Is)"
