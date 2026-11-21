#!/bin/bash
set -euo pipefail
r=$1
action=$2
export PYTHONPATH="$r/workflow:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2/dependencies:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2-public/deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
config=(--protocol "$r/workflow/reports/r19_plan/T4_PROTOCOL.md" --access "$r/access" --pin "$r/lock/access.json")
case "$action" in
predict)
  key=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$r/lock/keys")
  test -n "$key"
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_workflow predict "${config[@]}" --slice "$key" --out "$r/forecasts/$key"
  ;;
freeze)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_workflow freeze "${config[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" > "$r/freeze.sha256"
  ;;
score)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_workflow score "${config[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" --expected-sha256 "$(cat "$r/freeze.sha256")" --out "$r/score"
  ;;
esac
