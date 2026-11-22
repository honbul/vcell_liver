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
original="$plan_root/x-a.KfInaHqK"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
t2="$base/vcell_liver_goal_tracks/v1/t2"
report="$repo/reports/r19_plan"
x3="$plan_root/x3-b.zofrXFyk/donors"
export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/dependencies:$t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

common=(--access "$original/access" --pin "$report/data/x_a/access/receipt.json" --inputs "$original/inputs"
        --basal "$original/basal" --protocol "$report/XB_PROTOCOL.md" --new-donors "$run/donors"
        --hepg2-donors "$x3" --config "$base/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml"
        --assets "$base/vcell_liver_r15_lincs/v1/model-inputs" --forecasts "$run/forecasts" --freeze "$run/freeze.json")
cells=(HEPG2 A375 A549 HT29 MCF7)
roles=(A B C)
case "$phase" in
  donors|predict)
    task=$SLURM_ARRAY_TASK_ID
    if [[ "$phase" == donors ]]; then
      task=$((task + 9))
    fi
    slice="${cells[task/9]}-f$((task%9/3))-${roles[task%3]}"
    "$python" -m src.r19_plan.x_b "$phase" "${common[@]}" --slice "$slice" --out "$run/donors/$slice"
    ;;
  freeze)
    "$python" -m src.r19_plan.x_b freeze "${common[@]}"
    sha256sum "$run/freeze.json" | cut -d' ' -f1 > "$run/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.x_b score "${common[@]}" --expected-sha256 "$(cat "$run/freeze.sha256")" --out "$run/score"
    ;;
  *) exit 2 ;;
esac
