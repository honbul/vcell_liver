#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
assets="$base/vcell_liver_r11_chemical_probe/assets"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
args=(--reports "$repo/reports" --protocol "$repo/reports/r19_plan/HCC_FIXED44_MULTITASK_PROTOCOL.md"
  --previous "$root/hcc-scaffold-refit.lWd6KCZt" --panel-lock "$root/hcc-vehicle-panel.MvfmPZOH/lock"
  --reference "$root/hcc-fixed44-refit.WBfRgClA" --run "$run" --features "$assets/features/features.npz")
case "$phase" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    modes=(purged full)
    roles=(A B)
    args+=(--mode "${modes[task/6]}" --fold "$((task%6/2))" --role "${roles[task%2]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$assets/study-env/bin/python" -m src.r19_plan.hcc_fixed44_multitask "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
