#!/usr/bin/env bash
#SBATCH -J r2_hcc
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# The only job that sets R2_FINAL_EVAL. hcc_eval.py still refuses unless the freeze manifest is committed.
set -euo pipefail
source experiments/episodic_residual_r2/env.sh
export R2_FINAL_EVAL=1
echo "start=$(date -Is) commit=$R2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m src.episodic_residual_r2.hcc_eval "$@"
echo "end=$(date -Is)"
