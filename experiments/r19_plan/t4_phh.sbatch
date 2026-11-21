#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
set -euo pipefail

run=$1
phase=$2
plan_root=$(dirname "$run")
workbench=$(dirname "$plan_root")
repo="$run/repo"
python="$workbench/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
dependencies="$workbench/vcell_liver_goal_tracks/v1"
source_t4="$plan_root/t4-run.fS2rpsIm"
matrix_index="$plan_root/matrix-index.91pT0e4H/index"
reports="$repo/reports"
export PYTHONPATH="$repo:$dependencies/t2/dependencies:$dependencies/t2-public/deps"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
access=(--access "$run/access" --pin "$run/lock/access.json")
protocol="$reports/r19_plan/T4_PHH_PROTOCOL.md"

case "$phase" in
  metadata)
    "$python" -m src.r19_plan.t4_access --cell PHH --base "$workbench" --reports "$reports" \
      --r18-registered "$plan_root/audit.YBfVmbuo/r18-registered" \
      --r18-lock "$plan_root/preflight.kwjS2aLo/r18-input-lock" \
      --r18-access "$plan_root/access-plan.h0O8ex4z/allowlists" --r18-index "$matrix_index" \
      --support-reference "$reports/r19_plan/data/t4_support/receipt.json" \
      --matrix-reference "$reports/r19_plan/data/preflight/r18/index/GSE92742/receipt.json" \
      --landmarks "$source_t4/access/landmarks.tsv" --out "$run/access"
    ;;
  predict)
    key=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$run/lock/keys")
    test -n "$key"
    "$python" -m src.r19_plan.t4_workflow predict "${access[@]}" --protocol "$protocol" \
      --slice "$key" --out "$run/forecasts/$key"
    ;;
  freeze)
    "$python" -m src.r19_plan.t4_workflow freeze "${access[@]}" --protocol "$protocol" \
      --forecasts "$run/forecasts" --freeze "$run/freeze.json" > "$run/freeze.sha256"
    ;;
  score)
    "$python" -m src.r19_plan.t4_workflow score "${access[@]}" --protocol "$protocol" \
      --forecasts "$run/forecasts" --freeze "$run/freeze.json" \
      --expected-sha256 "$(cat "$run/freeze.sha256")" --out "$run/score"
    ;;
  *) exit 2 ;;
esac
