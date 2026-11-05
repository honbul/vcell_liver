#!/usr/bin/env bash
#SBATCH --job-name=rf_checkpoint_check
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --output=logs/paper_checkpoint_check_%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
work="${1:?provide a new output directory}"
.venvs/gears/bin/python -u src/checkpoint_inspect.py --track R1 --run-id R1-paper-public-v1-s1 \
    --output "$work/gears_checkpoint.json"
.venvs/scgpt/bin/python -u src/checkpoint_inspect.py --track R2 --run-id R2-manuscript-public-v1 \
    --output "$work/scgpt_checkpoint.json"
