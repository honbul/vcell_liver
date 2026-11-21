#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail

run=$1
phase=$2
plan_root=$(dirname "$run")
base=$(dirname "$plan_root")
repo="$run/repo"
x="$plan_root/x-a.KfInaHqK"
reports="$repo/reports/r19_plan"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

access=(--access "$x/access" --pin "$reports/data/x_a/access/receipt.json" --protocol "$reports/TA_PROTOCOL.md")
case "$phase" in
  audit)
    "$python" -m src.r19_plan.t_a_scope "${access[@]}" --out "$run/audit"
    exit 0
    ;;
esac
common=("${access[@]}" --audit "$run/audit" --audit-pin "$run/lock/audit.json" --inputs "$x/inputs"
  --config "$base/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml" --assets "$base/vcell_liver_r15_lincs/v1/model-inputs"
  --original "$x/freeze.json" --original-pin "$reports/data/x_a/freeze.json" --original-forecasts "$x/forecasts"
  --forecasts "$run/forecasts" --freeze "$run/freeze.json")
case "$phase" in
  predict)
    task=$SLURM_ARRAY_TASK_ID
    cells=(HEPG2 A375 A549 HT29 MCF7)
    roles=(A B C)
    key="${cells[task/9]}-f$((task%9/3))-${roles[task%3]}"
    "$python" -m src.r19_plan.t_a predict "${common[@]}" --slice "$key"
    ;;
  freeze)
    "$python" -m src.r19_plan.t_a freeze "${common[@]}"
    sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.t_a score "${common[@]}" --expected-sha256 "$(cat "$run/freeze.sha256")" --out "$run/score"
    ;;
  *) exit 2 ;;
esac
