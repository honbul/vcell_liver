#!/usr/bin/env bash
#SBATCH --job-name=r1_py
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r1_py_%j.log
# Run one R1 script (diagnostics, summaries, evaluation).
# Submit: SCRIPT=<file.py> ARGS="<arguments>" sbatch experiments/softprior_multitask_r1/py.sbatch   (environment is exported; --export would split ARGS at commas)
set -euo pipefail
: "${SCRIPT:?}"
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) script=$SCRIPT args=${ARGS:-} start=$(date -Is)"
# shellcheck disable=SC2086
"$R1_PY" "$R1_SRC/$SCRIPT" ${ARGS:-}
echo "end=$(date -Is)"
