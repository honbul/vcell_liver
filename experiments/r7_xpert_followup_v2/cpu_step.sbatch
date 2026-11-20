#!/usr/bin/env bash
#SBATCH -J r7v2_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r7v2_<name> experiments/r7_xpert_followup_v2/cpu_step.sbatch <py:xp|tb> <module> [args]
set -euo pipefail
source experiments/r7_xpert_followup_v2/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
PY=$XP_PY; [ "$1" = tb ] && PY=$TB_PY
echo "start=$(date -Is) commit=$R7V2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
TF="$TMPDIR/time_${SLURM_JOB_ID:-x}.txt"
/usr/bin/time -v -o "$TF" "$PY" -m "src.r7_xpert_followup_v2.$2" "${@:3}"
grep -E "Maximum resident|Elapsed|User time|System time" "$TF" || true
echo "end=$(date -Is)"
