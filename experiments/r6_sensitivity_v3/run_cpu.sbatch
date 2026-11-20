#!/usr/bin/env bash
# Generic CPU job of the R6 v3 track. Submit ONLY through the cycle wrapper:
#   /mnt/S1/sdata/workbench/Virtual_Cell/gssong/v3_tools/v3_sbatch.sh --job-name=v3r6_<name> --cpus-per-task=<=4> \
#       --mem=<..> --time=<..> experiments/r6_sensitivity_v3/run_cpu.sbatch <dotted module> [args...]
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/r6_sensitivity_v3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) dirty=$(git status --porcelain -uno | wc -l) start=$(date -Is) args=$*"
MOD=$1; shift
if [ -x /usr/bin/time ]; then /usr/bin/time -v "$R6_PY" -u -m "$MOD" "$@"; else "$R6_PY" -u -m "$MOD" "$@"; fi
echo "end=$(date -Is)"
