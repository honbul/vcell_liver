#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

run=$1
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
public="$base/vcell_liver_goal_tracks/v1"
export PYTHONPATH="$repo:$public/t2/dependencies:$public/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
exec /usr/bin/time -v "$python" -m src.r19_plan.jhh_low_dose_support --base "$base" --reports "$repo/reports" \
  --r18-registered "$root/audit.YBfVmbuo/r18-registered" --r18-lock "$root/preflight.kwjS2aLo/r18-input-lock" \
  --r18-access "$root/access-plan.h0O8ex4z/allowlists" --r18-index "$root/matrix-index.91pT0e4H/index" \
  --ten "$root/hcc-lincs24.weD7cHPm/audit" \
  --protocol "$repo/reports/r19_plan/JHH_LOW_DOSE_SUPPORT_PROTOCOL.md" --out "$run/audit"
