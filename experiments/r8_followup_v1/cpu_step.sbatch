#!/usr/bin/env bash
#SBATCH -J r8f_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage (server, worktree root): sbatch -J r8f_<name> experiments/r8_followup_v1/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/r8_followup_v1/env.sh
echo "start=$(date -Is) commit=$R8F_COMMIT dirty=$(git status --porcelain -- src configs experiments tests | wc -l) host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m "$1" "${@:2}"
echo "end=$(date -Is)"
