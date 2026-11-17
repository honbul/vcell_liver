#!/usr/bin/env bash
#SBATCH -J r8_multi
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -c 12
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# usage: sbatch -J r8_<name> experiments/replicate_calibrated_r8/gpu_multi.sbatch <task file> [parallel=6]
# The task file has one "<module> [args]" per line; the small models share one GPU, <parallel> at a time. Each task logs to its own file.
set -uo pipefail
source experiments/replicate_calibrated_r8/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
P=${2:-6}
echo "start=$(date -Is) commit=$R8_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) tasks=$1 parallel=$P"
n=0; fail=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  n=$((n+1))
  ( "$TB_PY" -m src.replicate_calibrated_r8.$line > "assets/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_t${n}.out" 2>&1; echo "task $n rc=$? : $line" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do wait -n; done
done < "$1"
wait
echo "end=$(date -Is)"
