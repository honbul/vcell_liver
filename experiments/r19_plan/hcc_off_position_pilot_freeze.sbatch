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
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--run "$run" --protocol "$report/HCC_OFF_POSITION_PILOT_FREEZE_PROTOCOL.md"
  --pilot "$root/hcc-off-pilot.GzfiFSmB/audit" --pilot-pin "$report/data/hcc_off_position_pilot/audit/receipt.json"
  --previous "$root/jhh5-off-position.0ldmuIkJ" --previous-protocol "$report/JHH5_OFF_POSITION_PROTOCOL.md"
  --previous-access-pin "$report/data/jhh5_off_position/access/receipt.json"
  --previous-freeze-pin "$report/data/jhh5_off_position/freeze.json" --state "$public/upstream-state")
case "$phase" in
  predict)
    roles=(A B C)
    args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}" ) ;;
  access|freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_off_position_pilot_freeze "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
