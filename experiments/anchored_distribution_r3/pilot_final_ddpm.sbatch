#!/usr/bin/env bash
# Pilot, held-out stage (after the pilot freeze): retrain the two frozen DDPM candidates on all
# TRAIN drugs with 3 training seeds and score them once on the TEST drugs (3 sampling seeds).
#SBATCH --job-name=r3_pfin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
$PY.freeze_pilot --verify
for S in 0 1 2; do
  $PY.ddpm --setting final --target resid --scale resid_global --lr 3e-4 --steps 20000 --seed $S --stage pilot_final
  $PY.ddpm --setting final --target abs --lr 1e-3 --steps 20000 --seed $S --stage pilot_final
done
M=$(ls "$R3_ASSETS/runs/ddpm" | grep -E "^final-(resid-resid_global-lr0.0003|abs-std-lr0.001)-film-w256b2-st20000-s[012]-" | sed 's/^/ddpm:/' | paste -sd,)
echo "methods: $M"
$PY.evaluate --setting final --tag pilot_final_ddpm --methods "$M" --sample-seeds 0,1,2
echo "end=$(date -Is)"
