#!/usr/bin/env bash
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage (server, worktree root), only through the v3 guard:
#   /mnt/S1/sdata/workbench/Virtual_Cell/gssong/v3_tools/v3_sbatch.sh --job-name=v3r8_<name> --cpus-per-task=4 --mem=24G \
#       --time=04:00:00 experiments/r8_unseen_drug_v3/cpu_step.sbatch <module> [args]
# closed-form CPU only (GPU 0).
set -euo pipefail
source experiments/r8_unseen_drug_v3/env.sh
echo "start=$(date -Is) commit=$R8V3_COMMIT dirty=$(git status --porcelain -- src configs experiments | wc -l) host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
/usr/bin/time -v -o "assets/logs/time_${SLURM_JOB_ID:-local}.txt" "$TB_PY" -m "$1" "${@:2}"
echo "end=$(date -Is)"
