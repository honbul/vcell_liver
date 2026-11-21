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
report="$repo/reports/r19_plan"
public="$base/vcell_liver_goal_tracks/v1/t2-public"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$public/state-deps:$public/deps:$base/vcell_liver_goal_tracks/v1/t2/dependencies"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(--run "$run" --protocol "$report/JHH7_DONOR_TRANSFER_PROTOCOL.md"
  --audit "$root/hcc-lincs24.weD7cHPm/audit" --audit-pin "$report/data/hcc_lincs24/audit/receipt.json"
  --cmap-pin "$report/data/preflight/r18/index/CMap2020/receipt.json"
  --access-pin "$report/data/jhh7_donor_transfer/access/receipt.json"
  --landmarks "$base/vcell_liver_r15_lincs/v1/cache/GSE92742/landmarks.tsv" --state "$public/upstream-state")
case "$phase" in
  predict)
    roles=(A B C)
    args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}")
    ;;
  score) args+=(--digest "$(cat "$run/freeze.sha256")") ;;
  lock|freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.jhh7_donor_transfer "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
