#!/usr/bin/env bash
#SBATCH -J r4_cpu
#SBATCH -p cpu
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r4_<name> experiments/response_geometry_r4/cpu_step.sbatch <module> [args]
set -euo pipefail
source experiments/response_geometry_r4/env.sh
echo "start=$(date -Is) commit=$R4_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} args=$*"
"$TB_PY" -m "src.response_geometry_r4.$1" "${@:2}"
echo "end=$(date -Is)"
