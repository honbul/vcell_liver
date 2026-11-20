#!/usr/bin/env bash
#SBATCH -J r2_cpu
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 04:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r2_<name> experiments/episodic_residual_r2/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/episodic_residual_r2/env.sh
echo "start=$(date -Is) commit=$R2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m "src.episodic_residual_r2.$1" "${@:2}"
echo "end=$(date -Is)"
