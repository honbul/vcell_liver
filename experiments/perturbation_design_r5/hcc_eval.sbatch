#!/usr/bin/env bash
#SBATCH -J r5_hcc
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem 96G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# The only job that sets R5_FINAL_EVAL. hcc.py still refuses unless the freeze manifest is committed at HEAD.
set -euo pipefail
source experiments/perturbation_design_r5/env.sh
export R5_NPROC=${SLURM_CPUS_PER_TASK:-8}
export R5_FINAL_EVAL=1
echo "start=$(date -Is) commit=$R5_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
test -z "$(git status --porcelain --untracked-files=no)" || { echo "server worktree not clean"; exit 3; }
"$TB_PY" -m src.perturbation_design_r5.hcc "$@"
echo "end=$(date -Is)"
