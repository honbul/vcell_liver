#!/bin/bash
set -euo pipefail
r=$1
action=$2
export PYTHONPATH="$r/repo:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2/dependencies:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_goal_tracks/v1/t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
args=(--protocol "$r/repo/reports/r19_plan/T4_HEPARG_PROTOCOL.md" --bridge "$r/bridge.json" --allowlists "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/t4-heparg-audit.OKFFgqbC/results")
args+=(--p3 "$r/repo/reports/p123/data/heparg/receipt.json" --genes "$r/repo/reports/p123/data/heparg/landmark_map.tsv")
args+=(--pin "$r/repo/reports/r19_plan/data/t4/access.json" --access "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/t4-run.fS2rpsIm/access" --original "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/t4-run.fS2rpsIm/freeze.json")
case "$action" in
predict)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_heparg_compare predict "${args[@]}" --original-forecasts "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r19_plan/t4-run.fS2rpsIm/forecasts" --out "$r/forecasts"
  ;;
freeze)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_heparg_compare freeze "${args[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" > "$r/freeze.sha256"
  ;;
score)
  /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r11_chemical_probe/assets/study-env/bin/python -m src.r19_plan.t4_heparg_compare score "${args[@]}" --forecasts "$r/forecasts" --freeze "$r/freeze.json" --expected-sha256 "$(cat "$r/freeze.sha256")" --heparg "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r14_cross_organ/execution-v1/data" --out "$r/score"
  ;;
esac
