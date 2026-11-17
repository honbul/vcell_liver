#!/usr/bin/env bash
#SBATCH -J r7_xpstep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 08:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r7_<name> experiments/evidence_gated_r7/xpert_step.sbatch <module> [args]   (XPert env)
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$XP_PY" -m "src.evidence_gated_r7.$1" "${@:2}"
echo "end=$(date -Is)"
