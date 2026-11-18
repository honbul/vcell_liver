#!/usr/bin/env bash
# Generic CPU job. Usage: sbatch -J r6f_<name> experiments/r6_followup_v1/run_cpu.sbatch <module> [args...]
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/r6_followup_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is) args=$*"
MOD=$1; shift
"$R6_PY" -u -m "src.r6_followup_v1.$MOD" "$@"
echo "end=$(date -Is)"
