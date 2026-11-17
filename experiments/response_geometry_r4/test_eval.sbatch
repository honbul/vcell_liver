#!/usr/bin/env bash
#SBATCH -J r4_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 04:00:00
#SBATCH -o assets/logs/%x_%j.out
# The only job that sets R4_FINAL_EVAL. test_eval.py still refuses unless the freeze manifest is committed and the tree is clean.
set -euo pipefail
source experiments/response_geometry_r4/env.sh
export R4_FINAL_EVAL=1
echo "start=$(date -Is) commit=$R4_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m src.response_geometry_r4.test_eval
echo "end=$(date -Is)"
