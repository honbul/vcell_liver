#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
reports="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
args=(--reports "$reports" --previous "$root/t4-run.fS2rpsIm" --protocol "$reports/T4_SCAFFOLD_PROTOCOL.md"
  --forecasts "$run/forecasts" --freeze "$run/freeze.json" --out "$run/score")
case "$phase" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    roles=(A B C)
    args+=(--group "f$((task / 3))-${roles[task%3]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -m src.r19_plan.t4_scaffold_transfer "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
