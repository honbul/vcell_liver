#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:25:00
set -euo pipefail

study=$1
phase=$2
base=$(dirname "$(dirname "$study")")
repo="$study/repo"
goal="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$goal/t2/dependencies:$goal/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
opts=(--protocol "$repo/reports/p4/PROTOCOL.md" --archive "$repo/reports" --prior "$goal"
      --p123 "$base/vcell_liver_p123/v1" --out "$base/vcell_liver_p4/v1"
      --study "$study" --design "$repo/reports/r19_plan/TAHOE_HCC_REPEAT_PROTOCOL.md")
if [[ "$phase" == audit ]]; then
  "$python" -m src.r19_plan.tahoe_hcc_repeat audit "${opts[@]}"
  exit 0
fi
doses=(0.5 5)
dose="${doses[SLURM_ARRAY_TASK_ID]}"
branch="$study/d$dose"
case "$phase" in
  lock)
    "$python" -m src.r19_plan.tahoe_hcc_repeat lock "${opts[@]}" --dose "$dose"
    sha256sum "$branch/freeze.json" | cut -d' ' -f1 > "$branch/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.tahoe_hcc_repeat score "${opts[@]}" --dose "$dose" \
      --expected-sha256 "$(cat "$branch/freeze.sha256")"
    ;;
  *) exit 2 ;;
esac
