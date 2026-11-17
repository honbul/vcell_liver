#!/usr/bin/env bash
#SBATCH -J r7_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r7_<name> experiments/evidence_gated_r7/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m "src.evidence_gated_r7.$1" "${@:2}"
echo "end=$(date -Is)"
