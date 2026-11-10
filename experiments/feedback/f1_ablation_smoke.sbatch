#!/usr/bin/env bash
#SBATCH --job-name=f1_abl_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=logs/f1_ablation_smoke_%j.log
# Smoke test every registered F1 ablation option: 2 epochs x N updates each, full validation.
# Outputs go to assets/runs/A-smoke/<run-id>/ (never counted as results).
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONHASHSEED=0
updates="${1:-20}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
ids=$(.venvs/scgpt/bin/python -c 'import json;print(" ".join(r["run_id"] for r in json.load(open("experiments/feedback/f1_ablation_runs.json"))["runs"]))')
for id in $ids; do
  echo "=== SMOKE $id"
  .venvs/scgpt/bin/python -u src/feedback/f1_ablation_train.py --run-id "$id" --smoke-updates "$updates" --smoke-epochs 2
done
echo SMOKE_ALL_DONE
