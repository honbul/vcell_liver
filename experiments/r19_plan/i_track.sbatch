#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
plan="$3"
views="$4"
base="$5"
prior="$base/vcell_liver_r15_lincs/v1"
r17="$base/vcell_liver_r17_lincs/v1"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
report=reports/r19_plan
pin="$report/data/preflight/i_track/receipt.json"
protocol="$report/I_VIEW_PROTOCOL.md"
workflow=(--plan "$plan" --split-reference "$pin" --cache "$prior/cache/GSE92742"
          --forecasts "$run/forecasts" --protocol "$protocol" --freeze "$run/freeze.json")

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

case "$phase" in
  stage|predict)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^[0-8]$ ]]
    roles=(A B C)
    slice="f$((task/3))-${roles[task%3]}"
    if [[ "$phase" == "stage" ]]; then
      exec "$python" -m src.r19_plan.i_stage --slice "$slice" --cache "$prior/cache/GSE92742" \
        --vehicles "$r17/vehicles" --plan "$plan" --pin "$pin" --out "$run/predict-inputs/$slice"
    fi
    exec "$python" -m src.r19_plan.i_run --slice "$slice" --bundle "$run/predict-inputs/$slice" \
      --split-reference "$pin" --views "$views" --view-references "$report/data/i_views" --assets "$prior/model-inputs" \
      --config "$base/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml" --protocol "$protocol" --out "$run/forecasts"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.i_workflow freeze "${workflow[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.i_workflow score "${workflow[@]}" --vehicles "$r17/vehicles" \
      --expected-sha256 "$digest" --out "$run/score"
    ;;
  *)
    printf '%s\n' "Unknown I-track phase $phase" >&2
    exit 2
    ;;
esac
