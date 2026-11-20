#!/usr/bin/env bash
# R3-0a: index the real single-cell sample (read-only source) into this study's cache.
#SBATCH --job-name=r3_cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r3_cache_%j.log
set -euo pipefail
source experiments/anchored_distribution_r3/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R3_PY" -u -m src.anchored_distribution_r3.prepare_cache \
  --h5ad "$R3_H5AD" --v2-data "$R3_V2DATA" --out "$R3_ASSETS/cache"
echo "end=$(date -Is)"
