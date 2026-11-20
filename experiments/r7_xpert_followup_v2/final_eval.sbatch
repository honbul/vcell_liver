#!/usr/bin/env bash
#SBATCH -J r7v2_final
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# Frozen R4-test evaluations (correction impact + section 4.2 step 6). Run once after the freeze and all trajectories.
set -euo pipefail
source experiments/r7_xpert_followup_v2/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
echo "start=$(date -Is) commit=$R7V2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
export R7V2_FINAL_EVAL=1 R7_FINAL_EVAL=1     # R7_FINAL_EVAL unlocks evidence_gated_r7.data.Store.test_delta (its own checks apply)
"$XP_PY" -m src.r7_xpert_followup_v2.final_eval
echo "end=$(date -Is)"
