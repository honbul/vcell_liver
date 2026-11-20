#!/usr/bin/env bash
# Generic single-GPU job. Usage: sbatch -J r6v2_<name> experiments/r6_specificity_v2/run_cpu.sbatch <module path> [args...]
# <module path> is a full dotted module (e.g. src.r6_specificity_v2.repeat_audit).
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/r6_specificity_v2/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) dirty=$(git status --porcelain -uno | wc -l) start=$(date -Is) args=$*"
MOD=$1; shift
if [ -x /usr/bin/time ]; then /usr/bin/time -v "$R6_PY" -u -m "$MOD" "$@"; else "$R6_PY" -u -m "$MOD" "$@"; fi
echo "end=$(date -Is)"
