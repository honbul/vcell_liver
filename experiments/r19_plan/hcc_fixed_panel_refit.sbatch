#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:35:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
assets="$base/vcell_liver_r11_chemical_probe/assets"
tahoe="$base/vcell_liver_goal_tracks/v1"
public="$tahoe/t2-public"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps:$tahoe/t2/dependencies"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--reports "$repo/reports" --protocol "$repo/reports/r19_plan/HCC_FIXED_PANEL_REFIT_PROTOCOL.md"
  --previous "$root/hcc-scaffold-refit.lWd6KCZt" --panel-lock "$root/hcc-vehicle-panel.MvfmPZOH/lock"
  --features "$assets/features/features.npz" --controls "$base/vcell_liver_r12_multiaxis/assets/doses/source"
  --state "$public/upstream-state" --cells "$tahoe/t2/context-v2/cells-v2" --run "$run")
case "$phase" in
  cpu|state)
    task=${SLURM_ARRAY_TASK_ID:?}
    modes=(purged full)
    roles=(A B)
    args+=(--lane "$phase" --mode "${modes[task/6]}" --fold "$((task%6/2))" --role "${roles[task%2]}")
    phase=predict
    ;;
  freeze) ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_fixed_panel_refit "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
