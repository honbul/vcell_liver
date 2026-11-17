#!/usr/bin/env bash
# User-approved post-gate pilot, development stage (TRAIN drugs only, dev lines, dev folds).
# Trains the pre-recorded grid (residual DDPM: 2 LR x 2 scale; absolute DDPM: 2 LR), then scores
# every run on dev fold 2 with the unchanged evaluation code and aggregates next to the baselines.
#SBATCH --job-name=r3_pdev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
STEPS=${STEPS:-20000}
for LR in 1e-3 3e-4; do
  for SC in resid_global ctrl_var; do
    $PY.ddpm --setting dev --target resid --scale $SC --lr $LR --steps $STEPS --seed 0 --stage pilot_dev
  done
  $PY.ddpm --setting dev --target abs --lr $LR --steps $STEPS --seed 0 --stage pilot_dev
done
M=$(ls "$R3_ASSETS/runs/ddpm" | grep "^dev-" | grep -- "-st${STEPS}-s0-" | sed 's/^/ddpm:/' | paste -sd,)
echo "methods: $M"
$PY.evaluate --setting dev --tag pilot_dev --methods "$M"
$PY.aggregate --setting dev --tags c1_base,pilot_dev --out reports/anchored_distribution_r3/tables/pilot_dev_grid \
  --refs shifted:anchored2,gauss_lr50:anchored2,control:native
echo "end=$(date -Is)"
