#!/usr/bin/env bash
# CellFlow v0.0.9 (external comparator): export the setting's cells once, train once, write native
# generated cells for the setting's evaluation wells.
#   sbatch --export=ALL,SETTING=dev,ITERS=50000 experiments/anchored_distribution_r3/cellflow.sbatch
# Optional: SEED (default 0), SMOKE=1 (short check; predicts ~24 wells only).
#SBATCH --job-name=r3_cf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
SETTING=${SETTING:?set SETTING=dev|final}
ITERS=${ITERS:?set ITERS}
CF_PY="$R3_ASSETS/envs/cellflow/bin/python"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) setting=$SETTING iters=$ITERS start=$(date -Is)"
if [ ! -f "$R3_ASSETS/cellflow/$SETTING/export.json" ]; then
  $R3_PY -u -m src.anchored_distribution_r3.cellflow_adapter export --setting "$SETTING"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
$CF_PY -u src/anchored_distribution_r3/cellflow_adapter.py train --setting "$SETTING" --iters "$ITERS" \
  --seed "${SEED:-0}" ${SMOKE:+--smoke}
echo "end=$(date -Is)"
