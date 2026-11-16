#!/usr/bin/env bash
#SBATCH -J tb_gstep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.out
# usage: sbatch -J tb_<name> gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
export TB_COMMIT=$(git rev-parse --short HEAD)
echo "start=$(date -Is) commit=$TB_COMMIT host=$(hostname) args=$*"
"$TB_PY" -m "src.transigen_adaptation_b1.$1" "${@:2}"
echo "end=$(date -Is)"
