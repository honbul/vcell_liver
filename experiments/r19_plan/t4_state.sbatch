#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
tahoe="$base/vcell_liver_goal_tracks/v1"
public="$tahoe/t2-public"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps:$tahoe/t2/dependencies"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--reports "$repo/reports/r19_plan" --protocol "$repo/reports/r19_plan/T4_STATE_PROTOCOL.md"
  --previous "$root/t4-run.fS2rpsIm" --scaffold "$root/t4-scaffold.f8xmUYT0" --state "$public/upstream-state" --run "$run")
case "$phase" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    roles=(A B C)
    args+=(--group "f$((task/3))-${roles[task%3]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.t4_state "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
