#!/usr/bin/env bash
#SBATCH --job-name=rf_common_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/common_runtime_check_%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
runtime="${1:?specify gears or scgpt as the first argument}"
shift
case "$runtime" in gears|scgpt) ;; *) exit 2;; esac
exec ".venvs/$runtime/bin/python" -u src/common_runtime_check.py --model "$runtime" "$@"
