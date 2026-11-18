#!/bin/bash
# Packed runner: one Slurm array element runs PACK task ids concurrently (8 threads each) to keep the number of
# queued jobs small. Usage:
#   sbatch -p cpu -c $((8*PACK)) -a 0-M%T run_pack.sh <PACK> <task_id_file> <script.py> [args...]
# task_id_file: one task id per line; element i runs lines i*PACK .. i*PACK+PACK-1.
set -uo pipefail
source /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state/scripts/r9/env.sh
PACK=$1; LIST=$2; shift 2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
i=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t IDS < <(sed -n "$((i*PACK+1)),$((i*PACK+PACK))p" "$LIST")
echo "[r9] host=$(hostname) job=${SLURM_JOB_ID:-na} element=$i ids=${IDS[*]} commit=$(git -C "$R9_ROOT" rev-parse HEAD) start=$(date -Is)"
pids=()
for t in "${IDS[@]}"; do
  ( export R9_TASK=$t SLURM_ARRAY_TASK_ID=$t; "$R9_PY" "$@" > "$R9_ASSETS/logs/pack_${SLURM_JOB_NAME:-r9}_task${t}.out" 2>&1; echo "[r9] task $t exit $?" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "[r9] end=$(date -Is)"
