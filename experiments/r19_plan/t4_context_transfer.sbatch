#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
report="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
task=${SLURM_ARRAY_TASK_ID:?}
context=$task
if [[ "$phase" == predict ]]; then
  context=$((task / 9))
fi
cells=(PHH HEPG2)
target=${cells[context]}
output="$run/$target"
args=(--target "$target" --reports "$report" --protocol "$report/T4_CONTEXT_TRANSFER_PROTOCOL.md"
  --hepg2 "$root/t4-run.fS2rpsIm" --phh "$root/t4-phh.p1AqqE6o"
  --forecasts "$output/forecasts" --freeze "$output/freeze.json")
case "$phase" in
  predict)
    roles=(A B C)
    group="f$((task%9/3))-${roles[task%3]}"
    /usr/bin/time -v "$python" -m src.r19_plan.t4_context_transfer predict "${args[@]}" --group "$group" --out "$output/forecasts/$group"
    ;;
  freeze)
    /usr/bin/time -v "$python" -m src.r19_plan.t4_context_transfer freeze "${args[@]}"
    sha256sum "$output/freeze.json" | cut -d' ' -f1 > "$output/freeze.sha256"
    ;;
  score)
    /usr/bin/time -v "$python" -m src.r19_plan.t4_context_transfer score "${args[@]}" --digest "$(cat "$output/freeze.sha256")" --out "$output/score"
    ;;
  *) exit 2 ;;
esac
