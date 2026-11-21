#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
audit=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
public="$base/vcell_liver_goal_tracks/v1"
report="$repo/reports/r19_plan"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/time -v "$python" -m src.r19_plan.hcc_lincs_24h_access --audit "$audit" \
  --audit-pin "$report/data/hcc_lincs24/audit/receipt.json" \
  --cmap-pin "$report/data/preflight/r18/index/CMap2020/receipt.json" \
  --landmarks "$base/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" \
  --protocol "$report/HCC_LINCS_24H_TRANSFER_PROTOCOL.md" --out "$run/access"
