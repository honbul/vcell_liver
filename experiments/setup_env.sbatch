#!/usr/bin/env bash
#SBATCH --job-name=rf_env
#SBATCH --partition=cpu
#SBATCH --nodelist=g1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=logs/env_%j.log
set -euo pipefail
bash experiments/setup_env.sh gears
bash experiments/setup_env.sh scgpt
