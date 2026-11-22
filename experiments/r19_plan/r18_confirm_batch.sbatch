#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase=$1
run=$2
case "$phase" in
  gpu) limit=432 ;;
  merge) limit=108 ;;
  *) exit 2 ;;
esac

# One allocation handles several fixed slices sequentially to fit the Slurm submit quota.
task="${SLURM_ARRAY_TASK_ID:?}"
[[ "$task" =~ ^(0|[1-9][0-9]*)$ ]]
start=$((task*12))
[[ "$start" -lt "$limit" ]]
status=0
for ((slice=start; slice<start+12 && slice<limit; slice++)); do
  if ! SLURM_ARRAY_TASK_ID="$slice" /usr/bin/time -v bash experiments/r19_plan/r18_confirm.sbatch "$@" \
      > "$run/logs/$phase-slice-$slice-${SLURM_ARRAY_JOB_ID:?}.log" 2>&1; then
    printf '%s failed for slice %s\n' "$phase" "$slice" >&2
    status=1
  fi
done
exit "$status"
