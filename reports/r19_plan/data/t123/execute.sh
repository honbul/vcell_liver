#!/bin/bash
set -euo pipefail
r=$1
action=$2
export PYTHONPATH="$r/workflow:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2/dependencies:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2-public/deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
config=(--protocol "$r/workflow/reports/r19_plan/T123_PROTOCOL.md" --access "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/x-a.KfInaHqK/access" --pin "$r/workflow/reports/r19_plan/data/x_a/access/receipt.json")
case "$action" in
predict)
  fold=$((SLURM_ARRAY_TASK_ID / 3))
  roles=(A B C)
  role=${roles[$((SLURM_ARRAY_TASK_ID % 3))]}
  key=HEPG2-f$fold-$role
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t123_workflow predict "${config[@]}" --slice "$key" --out "$r/forecasts/$key"
  ;;
freeze)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t123_workflow freeze "${config[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" > "$r/freeze.sha256"
  ;;
score)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t123_workflow score "${config[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" --expected-sha256 "$(cat "$r/freeze.sha256")" --out "$r/score"
  ;;
esac
