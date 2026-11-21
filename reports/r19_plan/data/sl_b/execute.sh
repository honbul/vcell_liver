#!/bin/bash
set -euo pipefail
r=$1
action=$2
export PYTHONPATH="$r/repo:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2/dependencies:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2-public/deps:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver/vcell_r12_runtime_cu126:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver/vcell_r12_runtime_transformers"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
args=(--protocol "$r/repo/reports/r19_plan/SL_B_PROTOCOL.md" --config "$r/repo/configs/r12_multiaxis/config.yaml")
args+=(--access "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/x-a.KfInaHqK/access" --pin "$r/repo/reports/r19_plan/data/x_a/access/receipt.json")
case "$action" in
predict)
  fold=$((SLURM_ARRAY_TASK_ID / 3))
  roles=(A B C)
  role=${roles[$((SLURM_ARRAY_TASK_ID % 3))]}
  key=HEPG2-f$fold-$role
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.sl_b predict "${args[@]}" --assets "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets" --slice "$key" --out "$r/forecasts/$key"
  ;;
freeze)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.sl_b freeze "${args[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" > "$r/freeze.sha256"
  ;;
score)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.sl_b score "${args[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" --expected-sha256 "$(cat "$r/freeze.sha256")" --out "$r/score"
  ;;
esac
