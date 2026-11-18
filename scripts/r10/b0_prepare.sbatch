#!/bin/bash
#SBATCH --job-name=r10_b0
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=assets/logs/%x_%j.out
set -euo pipefail
source scripts/r10/env.sh
/usr/bin/time -v $PY -P -m r10_dose_composition.b0_prepare --config configs/r10/r10_v1.json
$PY -P -m r10_dose_composition.split --config configs/r10/r10_v1.json --out assets/cache/r10_v1/drug_split.tsv
echo "end=$(date -Is)"
