#!/usr/bin/env bash
#SBATCH -J r7f_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem 48G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r7f_<name> experiments/r7_followup_v1/gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/r7_followup_v1/env.sh
git update-index -q --refresh >/dev/null 2>&1 || true; DIRTY=$(git status --porcelain --untracked-files=no); test -z "$DIRTY" || { echo "worktree not clean:"; echo "$DIRTY"; exit 3; }
echo "start=$(date -Is) commit=$R7F_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
"$TB_PY" -m "src.r7_followup_v1.$1" "${@:2}"
echo "end=$(date -Is)"
