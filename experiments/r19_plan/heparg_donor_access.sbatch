#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
run=$1
repo="$run/repo"
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
reports="$repo/reports"
plan="$reports/r19_plan"
dependencies="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$repo:$dependencies/t2/dependencies:$dependencies/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
exec /usr/bin/time -v "$python" -m src.r19_plan.heparg_donor_access \
  --base "$base" --reports "$reports" --r18-registered "$plan_root/audit.YBfVmbuo/r18-registered" \
  --r18-lock "$plan_root/preflight.kwjS2aLo/r18-input-lock" --r18-access "$plan_root/access-plan.h0O8ex4z/allowlists" \
  --r18-index "$plan_root/matrix-index.91pT0e4H/index" --matrix-reference "$plan/data/preflight/r18/index/GSE92742/receipt.json" \
  --support-reference "$plan/data/t4_support/receipt.json" --cache "$base/vcell_liver_r14_cross_organ/execution-v1" \
  --t3 "$base/vcell_liver_goal_tracks/v1/t3" --p3-score "$reports/p123/data/heparg/receipt.json" \
  --protocol "$plan/HEPARG_DONOR_PROTOCOL.md" --out "$run/audit"
