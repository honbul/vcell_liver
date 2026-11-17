#!/usr/bin/env bash
# Final fixed-step training on every source-train family: arms x seeds 42/43/44, (lr, steps) read from cv_selection.json.
# usage (server, repository root): bash experiments/replicate_calibrated_r8/submit_final.sh <tag> "<arms>"
set -euo pipefail
source experiments/replicate_calibrated_r8/env.sh
TAG=$1; ARMS=$2
SEL=assets/public/manifests/replicate_calibrated_r8/cv_selection.json
[ -e "$SEL" ] || SEL=manifests/replicate_calibrated_r8/cv_selection.json
mkdir -p assets/tasks
ALL=assets/tasks/final_${TAG}.txt
[ -e "$ALL" ] && { echo "task list $ALL exists: tags are immutable"; exit 1; }
for arm in $ARMS; do
  read -r LR STEPS < <("$TB_PY" -c "import json,sys; s=json.load(open('$SEL'))['selection']['$arm']; print(s['lr'], s['steps'])")
  for seed in 42 43 44; do
    echo "train --arm $arm --lr $LR --seed $seed --steps $STEPS --run-id final_${arm}_s${seed}_${TAG}" >> "$ALL"
  done
done
split -l 6 -d "$ALL" "assets/tasks/final_${TAG}_part"
for p in assets/tasks/final_${TAG}_part*; do sbatch -J r8_final_${TAG} experiments/replicate_calibrated_r8/cpu_multi.sbatch "$p" 6; done
