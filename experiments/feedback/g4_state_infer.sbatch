#!/usr/bin/env bash
# G4 task 1: State ST-HVG-Replogle zeroshot inference (held-out line controls only), CPU.
# Usage: sbatch experiments/feedback/g4_state_infer.sbatch <fold> [extra args]
#   GPU alternative: sbatch --partition=gpu --gres=gpu:1 experiments/feedback/g4_state_infer.sbatch <fold>
#SBATCH --job-name=g4_state_inf
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/g4_state_infer_%j.log
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PYTHONHASHSEED=0
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
PY=assets/envs/state/venv/bin/python
FOLD=${1:?fold}; shift
echo "host=$(hostname) job=${SLURM_JOB_ID:-} fold=$FOLD start=$(date -Is)"
"$PY" -u src/feedback/g4_state_infer.py --fold "$FOLD" "$@"
echo "end=$(date -Is)"
