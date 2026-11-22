#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
report="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1/t2-public"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps:$base/vcell_liver_goal_tracks/v1/t2/dependencies"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--run "$run" --reports "$report" --protocol "$report/HCC_DONOR8_STATE_PROTOCOL.md"
  --jhh5 "$root/jhh5-transfer.r0tiL2nJ" --attribution "$root/jhh5-donor-check.qZGK4C5u"
  --jhh7 "$root/jhh7-transfer.HRBqXsAJ" --panel "$root/hcc-panel-size.2g01ASvz" --state "$public/upstream-state")
case "$phase" in
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    roles=(A B C)
    if (( task < 9 )); then
      args+=(--group "JHH5-$((task/3))-${roles[task%3]}")
    else
      args+=(--group "JHH7-all-${roles[task%3]}")
    fi
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_donor8_state "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
