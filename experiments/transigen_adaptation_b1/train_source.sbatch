#!/usr/bin/env bash
#SBATCH -J tb_src
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 12:00:00
#SBATCH -o logs/tb_src_%x_%j.out
# usage: sbatch -J tb_src_<id> train_source.sbatch <lr> <seed> <run_id> [extra args]
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
export TB_COMMIT=$(git rev-parse --short HEAD)
echo "start=$(date -Is) commit=$TB_COMMIT host=$(hostname) args=$*"
"$TB_PY" -m src.transigen_adaptation_b1.train_source --lr "$1" --seed "$2" --run-id "$3" "${@:4}"
echo "end=$(date -Is)"
