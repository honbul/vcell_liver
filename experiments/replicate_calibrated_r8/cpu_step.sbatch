#!/usr/bin/env bash
#SBATCH -J r8_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 06:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r8_<name> experiments/replicate_calibrated_r8/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/replicate_calibrated_r8/env.sh
echo "start=$(date -Is) commit=$R8_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m "src.replicate_calibrated_r8.$1" "${@:2}"
echo "end=$(date -Is)"
