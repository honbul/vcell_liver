#!/usr/bin/env bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
code=$2
runs=$3
action=$4
location=$5
cd "$code"
r19="$study/vcell_liver_r19_plan"
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$study/vcell_liver_goal_tracks/v1/t2"
runtime="$study/vcell_liver/vcell_r12_runtime_cu126"
public="$t2-public"
export PYTHONPATH="$code:$runtime:$study/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps:$t2/dependencies"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
base=(--reports "$code/reports/r19_plan" --hepg2 "$r19/t4-run.fS2rpsIm" --phh "$r19/t4-phh.p1AqqE6o")
f=(--forecasts "$location/forecasts" --lock "$location/freeze.json")
case "$action" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    cells=(HEPG2 PHH)
    roles=(A B C)
    cell=${cells[task/9]}
    group="f$(((task%9)/3))-${roles[task%3]}"
    device=cpu
    if [[ "$cell" == PHH ]]; then device=cuda; fi
    "$python" -m tune_t4_v2.release predict "${base[@]}" --settings "$runs/rounds34-02/$cell/$group/receipt.json" \
      --cell "$cell" --group "$group" --device "$device" --pretrain "$runs/pretrain-01" \
      --hep-mean-shift "$r19/p1-rev-t4.qIhZDlyc" --hep-state "$r19/t4-state.Z5HjUCr5" \
      --repeat "$r19/t4-repeat.mUXzEDhP" --state-upstream "$public/upstream-state" \
      --out "$location/forecasts/$cell/$group"
    ;;
  freeze)
    "$python" -m tune_t4_v2.release freeze "${base[@]}" "${f[@]}"
    sha256sum "$location/freeze.json" | cut -d ' ' -f1 > "$location/freeze.sha256"
    ;;
  score)
    "$python" -m tune_t4_v2.release score "${base[@]}" "${f[@]}" --out "$location/score" \
      --digest "$(cat "$location/freeze.sha256")"
    ;;
  *) exit 2 ;;
esac
