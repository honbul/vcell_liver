#!/usr/bin/env bash
#SBATCH -J r7_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r7_<name> experiments/evidence_gated_r7/gpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
test -z "$(git status --porcelain --untracked-files=no)" || { echo "worktree not clean"; exit 3; }
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
"$TB_PY" -m "src.evidence_gated_r7.$1" "${@:2}"
echo "end=$(date -Is)"
