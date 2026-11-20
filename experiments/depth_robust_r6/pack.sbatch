#!/usr/bin/env bash
# Several small R6 runs sharing ONE GPU. Submit as an array with a concurrency cap so that the
# study never holds more than two GPUs:
#   sbatch -J r6_<name> --array=0-<n>%2 experiments/depth_robust_r6/pack.sbatch <task file> [runs per GPU; 2 on a 24 GB card]
# Each line of the task file is "<module> <args...>" (must contain --run-id <id>); array task i runs
# lines [i*PER, (i+1)*PER) concurrently and writes logs/run_<id>.log for each.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.log
set -uo pipefail
source experiments/depth_robust_r6/env.sh
TASKS=$1; PER=${2:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is) tasks=$TASKS per=$PER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" || true
first=$(( ${SLURM_ARRAY_TASK_ID:-0} * PER + 1 ))
mapfile -t LINES < <(sed -n "${first},$(( first + PER - 1 ))p" "$TASKS")
pids=(); ids=()
for line in "${LINES[@]}"; do
  [ -n "$line" ] || continue
  id=$(echo "$line" | sed -n 's/.*--run-id \([^ ]*\).*/\1/p')
  mod=${line%% *}; args=${line#* }
  echo "launch $id: $line"
  # shellcheck disable=SC2086
  "$R6_PY" -u -m "src.depth_robust_r6.$mod" $args > "logs/run_${id}.log" 2>&1 &
  pids+=($!); ids+=("$id")
done
rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "ok ${ids[$i]}"; else echo "FAILED ${ids[$i]}"; rc=1; fi
done
echo "end=$(date -Is) rc=$rc"
exit $rc
