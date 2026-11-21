#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

run=$1
phase=$2
base=$(dirname "$(dirname "$run")")
repo="$run/repo"
goal="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$goal/t2/dependencies:$goal/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
args=(--run "$run" --stage "$goal/t1-cache" --archive "$repo/reports"
      --protocol "$repo/reports/r19_plan/HCC_CROSS_DOSE_DONOR_PROTOCOL.md")
case "$phase" in
  audit)
    exec /usr/bin/time -v "$python" -u -m src.r19_plan.hcc_cross_dose_donor audit "${args[@]}"
    ;;
  diagnosis)
    audit_root=$3
    exec /usr/bin/time -v "$python" -u -m src.r19_plan.hcc_cross_dose_support_diagnosis "${args[@]}" \
      --audit "$audit_root" --out "$run/diagnosis"
    ;;
esac
exit 2
