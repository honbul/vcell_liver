#!/usr/bin/env bash
#SBATCH --job-name=rf_frozen_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=logs/common_test_%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
runtime="${1:?specify gears or scgpt as the first argument}"
shift
case "$runtime" in gears|scgpt) ;; *) exit 2;; esac
exec ".venvs/$runtime/bin/python" -u src/common_test.py "$@"
