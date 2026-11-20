#!/usr/bin/env bash
# R3-1 (dev): fit the chosen mean predictor on dev folds {0,1}, then score the distribution
# baselines that share it on dev fold 2 of the eight development cell lines.
#SBATCH --job-name=r31_dev
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r3_%x_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
PY="$R3_PY -u -m src.anchored_distribution_r3"
$PY.mean_model --mode fit --setting dev
$PY.evaluate --setting dev --tag r31_base --methods control,meanpoint,shifted,shifted_pool,gauss_diag,gauss_lr10,gauss_lr50
$PY.aggregate --setting dev --tags r31_base --out reports/anchored_distribution_r3/tables/dev_r31_baselines \
  --refs shifted:anchored,gauss_diag:anchored
echo "end=$(date -Is)"
