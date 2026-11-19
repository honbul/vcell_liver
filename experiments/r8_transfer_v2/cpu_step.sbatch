#!/usr/bin/env bash
#SBATCH -J r8v2_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage (server, worktree root): sbatch -J r8v2_<name> experiments/r8_transfer_v2/cpu_step.sbatch <module> [args]
# closed-form CPU only (GPU cap of this track = 0); at most 6 r8v2_ jobs at a time.
set -euo pipefail
source experiments/r8_transfer_v2/env.sh
echo "start=$(date -Is) commit=$R8V2_COMMIT dirty=$(git status --porcelain -- src configs experiments tests | wc -l) host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
/usr/bin/time -v -o "assets/logs/time_${SLURM_JOB_ID:-local}.txt" "$TB_PY" -m "$1" "${@:2}"
echo "end=$(date -Is)"
