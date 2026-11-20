#!/bin/bash
#SBATCH --job-name=r10_atlas
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=assets/logs/%x_%A_%a.out
set -euo pipefail
source scripts/r10/env.sh
LINES=(A549 K562 MCF7)
/usr/bin/time -v $PY -P -m r10_dose_composition.atlas --config configs/r10/r10_v1.json \
  --split assets/cache/r10_v1/drug_split.tsv --line ${LINES[$SLURM_ARRAY_TASK_ID]}
echo "end=$(date -Is)"
