#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/time -v "$python" -m src.r19_plan.hcc_time_lincs --base "$base" --reports "$repo/reports" \
  --r18-registered "$root/audit.YBfVmbuo/r18-registered" --r18-lock "$root/preflight.kwjS2aLo/r18-input-lock" \
  --r18-access "$root/access-plan.h0O8ex4z/allowlists" --r18-index "$root/matrix-index.91pT0e4H/index" \
  --protocol "$repo/reports/r19_plan/HCC_TIME_LINCS_PROTOCOL.md" --out "$run/audit"
