#!/usr/bin/env bash
#SBATCH --job-name=rf_model_selection
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/common_model_handoff_%j.log
set -euo pipefail
work="${1:?provide the common workspace}"
dataset="${2:?provide the prepared common dataset}"
exec .venvs/gears/bin/python -u src/common_selection.py model --dataset "$dataset" \
    --output "$work/model_selection.json"
