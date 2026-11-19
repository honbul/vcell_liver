#!/usr/bin/env bash
#SBATCH -J r7v2_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 72:00:00
#SBATCH -o assets/logs/%x_%A_%a.out
# usage: sbatch -J r7v2_<name> --array=0-<W-1>%<W> experiments/r7_xpert_followup_v2/gpu_worker.sbatch <queue.txt> [xtrain args]
# Each worker claims the next unclaimed trajectory of the queue (atomic mkdir) and trains it; stops claiming if the
# code/config trees of this track change under it (frozen code for the whole queue).
set -euo pipefail
source experiments/r7_xpert_followup_v2/env.sh
Q=$1; shift
tree() { echo "$(git rev-parse HEAD:src/r7_xpert_followup_v2) $(git rev-parse HEAD:configs/r7_xpert_followup_v2) $(git rev-parse HEAD:experiments/r7_xpert_followup_v2)"; }
T0=$(tree)
CL="$R7V2_ASSETS/claims/$(basename "$Q" .txt)"; mkdir -p "$CL"
echo "worker start=$(date -Is) commit=$R7V2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
while read -r TRAJ; do
  [ -z "$TRAJ" ] && continue
  [ "$(tree)" = "$T0" ] || { echo "code tree changed under the worker; stop claiming"; break; }
  git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean: $DIRTY"; break; }
  mkdir "$CL/$TRAJ" 2>/dev/null || continue
  echo "$(date -Is) job=${SLURM_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) commit=$(git rev-parse --short HEAD) claim $TRAJ args=$*" | tee "$CL/$TRAJ/claim.txt"
  if "$XP_PY" -m src.r7_xpert_followup_v2.xtrain --traj "$TRAJ" "$@" > "$R7V2_ASSETS/logs/traj_${TRAJ}$(echo "$*" | grep -o -- '--run-suffix [^ ]*' | awk '{print $2}').log" 2>&1; then
    echo ok > "$CL/$TRAJ/status"; echo "$(date -Is) done $TRAJ"
  else
    echo fail > "$CL/$TRAJ/status"; echo "$(date -Is) FAILED $TRAJ"
  fi
done < "$Q"
echo "worker end=$(date -Is)"
