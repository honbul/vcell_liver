#!/usr/bin/env bash
# Inner family cross-validation: arms x learning rates x 3 folds, seed 42. 5 tasks per CPU job (the models are small).
# usage (server, repository root): bash experiments/replicate_calibrated_r8/submit_cv.sh <tag> "<arms>" "<lrs>" [extra train args]
set -euo pipefail
source experiments/replicate_calibrated_r8/env.sh
TAG=$1; ARMS=$2; LRS=$3; EXTRA=${4:-}
mkdir -p assets/tasks
ALL=assets/tasks/cv_${TAG}.txt
[ -e "$ALL" ] && { echo "task list $ALL exists: tags are immutable"; exit 1; }
for arm in $ARMS; do for lr in $LRS; do for f in 0 1 2; do
  echo "train --arm $arm --lr $lr --seed 42 --fold $f --run-id cv_${arm}_lr${lr}_f${f}_${TAG} $EXTRA" >> "$ALL"
done; done; done
split -l 5 -d "$ALL" "assets/tasks/cv_${TAG}_part"
for p in assets/tasks/cv_${TAG}_part*; do sbatch -J r8_cv_${TAG} experiments/replicate_calibrated_r8/cpu_multi.sbatch "$p" 5; done
