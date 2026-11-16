#!/usr/bin/env bash
#SBATCH -J tb_build
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 06:00:00
#SBATCH -o logs/tb_build_%j.out
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
echo "start=$(date -Is) commit=$(git rev-parse --short HEAD) host=$(hostname)"
"$TB_PY" -m src.transigen_adaptation_b1.build_dataset --raw "$TB_RAW" --out "$TB_PROC/cmap2020_v1"
echo "end=$(date -Is)"
