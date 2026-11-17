#!/usr/bin/env bash
#SBATCH --job-name=r1_grid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r1_grid_%A_%a.log
# Inner-CV loss tables for one grid. Array tasks split the (training set, prior) units.
# Submit: sbatch --array=0-(N-1) --export=ALL,RUN_ID=<immutable id>,GRID=<name>,SETS=internal|external,PRIORS=real[,perm101..],NT=N[,OUTER=1] experiments/softprior_multitask_r1/grid.sbatch
set -euo pipefail
: "${RUN_ID:?}" "${GRID:?}" "${SETS:?}" "${NT:?}"
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) run=$RUN_ID grid=$GRID sets=$SETS task=$SLURM_ARRAY_TASK_ID/$NT outer=${OUTER:-0} start=$(date -Is)"
"$R1_PY" "$R1_SRC/run_grid.py" --run-dir "$R1_RUNS/$RUN_ID" --grid "$GRID" --sets "$SETS" --priors "${PRIORS:-real}" \
  --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks "$NT" ${OUTER:+--outer-predict}
echo "end=$(date -Is)"
