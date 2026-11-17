#!/usr/bin/env bash
# R6 contract tests (CPU).
#SBATCH --job-name=r6_tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/depth_robust_r6/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R6_PY" -m unittest -v tests.depth_robust_r6.test_r6
echo "end=$(date -Is)"
