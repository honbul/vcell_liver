#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
task=$SLURM_ARRAY_TASK_ID
context=$task
if [[ "$phase" == predict ]]; then
  context=$((task / 9))
fi
cells=(HEPG2 PHH)
previous=(t4-run.fS2rpsIm t4-phh.p1AqqE6o)
protocols=(T4_PROTOCOL.md T4_PHH_PROTOCOL.md)
reports=(t4 t4_phh)
path="$run/${cells[context]}"
old="$plan_root/${previous[context]}"
report="$repo/reports/r19_plan"
options=(--protocol "$report/T4_REPEAT_PROTOCOL.md" --original-protocol "$report/${protocols[context]}"
  --access "$old/access" --pin "$old/lock/access.json" --original "$old/freeze.json" --original-forecasts "$old/forecasts"
  --published "$report/data/${reports[context]}/freeze.json" --forecasts "$path/forecasts" --freeze "$path/freeze.json")
case "$phase" in
  predict)
    roles=(A B C)
    group="f$((task%9/3))-${roles[task%3]}"
    "$python" -m src.r19_plan.t4_repeat predict "${options[@]}" --group "$group" --out "$path/forecasts/$group"
    ;;
  freeze)
    "$python" -m src.r19_plan.t4_repeat freeze "${options[@]}"
    sha256sum "$path/freeze.json" | cut -d' ' -f1 > "$path/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.t4_repeat score "${options[@]}" --expected-sha256 "$(cat "$path/freeze.sha256")" --out "$path/score"
    ;;
  *) exit 2 ;;
esac
