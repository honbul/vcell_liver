#!/usr/bin/env bash
#SBATCH -J tb_adapt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
# usage: sbatch -J tb_adapt_<target>_<phase> adapt.sbatch <target> <search|seeds|curve>
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
export TB_COMMIT=$(git rev-parse --short HEAD)
echo "start=$(date -Is) commit=$TB_COMMIT host=$(hostname) args=$*"
"$TB_PY" -m src.transigen_adaptation_b1.adapt_target --target "$1" --phase "$2"
echo "end=$(date -Is)"
