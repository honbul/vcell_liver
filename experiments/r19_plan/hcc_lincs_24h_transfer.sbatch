#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
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
args=(--access "$run/access" --access-pin "$report/data/hcc_lincs24/access/receipt.json"
  --protocol "$report/HCC_LINCS_24H_TRANSFER_PROTOCOL.md" --state "$public/upstream-state" --run "$run")
roles=(A B C)
case "$phase" in
  donor) args+=(--group "${roles[${SLURM_ARRAY_TASK_ID:?}]}") ;;
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    args+=(--group "f$((task/3))-${roles[task%3]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_lincs_24h_transfer "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
