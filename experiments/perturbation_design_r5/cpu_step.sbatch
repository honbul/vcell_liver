#!/usr/bin/env bash
#SBATCH -J r5_cpu
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem 96G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r5_<name> experiments/perturbation_design_r5/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/perturbation_design_r5/env.sh
export R5_NPROC=${SLURM_CPUS_PER_TASK:-8}
echo "start=$(date -Is) commit=$R5_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
test -z "$(git status --porcelain --untracked-files=no)" || { echo "server worktree not clean"; exit 3; }
"$TB_PY" -m "src.perturbation_design_r5.$1" "${@:2}"
echo "end=$(date -Is)"
