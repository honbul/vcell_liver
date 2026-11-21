#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail
run=$1
repo="$run/repo"
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
assets="$base/vcell_liver_r11_chemical_probe/assets"
deps="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$repo:$deps/t2/dependencies:$deps/t2-public/deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
pub=/mnt/S1/sdata/pubdata
exec /usr/bin/time -v "$assets/study-env/bin/python" -m src.r19_plan.hcc_pilot_chemistry \
  --base "$base" --archive "$repo/reports" --lincs "$pub/LINCS" --tahoe "$pub/Tahoe/Tahoe-100M" \
  --emerald "$pub/Virtual_Cell/EmeraldBay" --r18 "$plan_root/audit.YBfVmbuo/r18-registered" \
  --prism "$base/vcell_liver_dev_i/coverage.bk00jUHu/data/prism/primary-screen-replicate-collapsed-treatment-info.csv" \
  --layout "$repo/reports/r19_plan/data/hcc_pilot_layout/plate_layout.tsv" --protocol "$repo/reports/r19_plan/HCC_PILOT_CHEMISTRY_PROTOCOL.md" --out "$run/audit"
