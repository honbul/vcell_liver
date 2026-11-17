#!/usr/bin/env bash
# Cycle 1 (dev): restoration v1 vs v2 on the shared-mean baselines, plus mean-kind contrasts.
#SBATCH --job-name=r3_c1dev
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
$PY.evaluate --setting dev --tag c1_base \
  --methods control,shifted,shifted@zero,shifted@delta_linemean,shifted_pool,gauss_diag,gauss_lr50
$PY.aggregate --setting dev --tags c1_base --out reports/anchored_distribution_r3/tables/dev_c1_restoration \
  --refs shifted:anchored,shifted:anchored2,shifted:pre,gauss_lr50:anchored,gauss_lr50:anchored2
echo "end=$(date -Is)"
