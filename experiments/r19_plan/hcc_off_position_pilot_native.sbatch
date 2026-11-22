#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
set -euo pipefail

run=$1
phase=$2
root=$(dirname "$run")
base=$(dirname "$root")
repo="$run/repo"
paper="$repo/reports/r19_plan"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(--run "$run" --protocol "$paper/HCC_OFF_POSITION_PILOT_NATIVE_PROTOCOL.md"
  --forecast "$root/hcc-off-pilot-freeze.IhWnfPBE" --forecast-protocol "$paper/HCC_OFF_POSITION_PILOT_FREEZE_PROTOCOL.md"
  --forecast-pin "$paper/data/hcc_off_position_pilot/forecast/freeze.json"
  --pilot "$root/hcc-off-pilot.GzfiFSmB/audit" --pilot-pin "$paper/data/hcc_off_position_pilot/audit/receipt.json"
  --previous "$root/jhh5-off-position.0ldmuIkJ" --previous-protocol "$paper/JHH5_OFF_POSITION_PROTOCOL.md"
  --previous-access-pin "$paper/data/jhh5_off_position/access/receipt.json"
  --previous-freeze-pin "$paper/data/jhh5_off_position/freeze.json"
  --prnet "$t2/t2-public/upstream-prnet" --gene-map "$base/vcell_liver_r15_lincs/v1/model-inputs/metadata/genes.tsv")
roles=(A B C)
models=(XPert_morgan PRnet-well-adaptation chemCPA_P0 TranSiGen_Tahoe)
case "$phase" in
  prepare) args+=(--role "${roles[${SLURM_ARRAY_TASK_ID:?}]}" ) ;;
  predict)
    task=${SLURM_ARRAY_TASK_ID:?}
    args+=(--role "${roles[task%3]}" --model "${models[task/3]}") ;;
  freeze) ;;
  *) exit 2 ;;
esac
/usr/bin/time -v "$python" -u -m src.r19_plan.hcc_off_position_pilot_native "$phase" "${args[@]}"
if [[ "$phase" == freeze ]]; then
  sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
fi
