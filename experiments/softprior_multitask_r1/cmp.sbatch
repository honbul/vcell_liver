#!/usr/bin/env bash
#SBATCH --job-name=r1_cmp
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/r1_cmp_%A_%a.log
# Comparator models (one array task per group of training sets).
# Submit: sbatch --array=0-(N-1) --export=ALL,RUN_ID=<id>,SCRIPT=run_enet.py,SETS=internal|external,NT=N experiments/softprior_multitask_r1/cmp.sbatch
set -euo pipefail
: "${RUN_ID:?}" "${SCRIPT:?}" "${SETS:?}" "${NT:?}"
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=$RUN_ID script=$SCRIPT sets=$SETS task=$SLURM_ARRAY_TASK_ID/$NT start=$(date -Is)"
"$R1_PY" "$R1_SRC/$SCRIPT" --run-dir "$R1_RUNS/$RUN_ID" --sets "$SETS" --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks "$NT" ${EXTRA_ARGS:-}
echo "end=$(date -Is)"
