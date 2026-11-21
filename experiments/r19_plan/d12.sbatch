#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
access="$3"
base="$4"
old="$base/vcell_liver_dev_i/efficacy.wFFiyzRF"
old_plan="$base/vcell_liver_dev_i/i-preflight.TE8L9Fqo/locked"
r15="$base/vcell_liver_r15_lincs/v1"
r17="$base/vcell_liver_r17_lincs/v1"
t2="$base/vcell_liver_goal_tracks/v1"
r12="$base/vcell_liver_r12_multiaxis"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
report=reports/r19_plan
old_pin="$report/data/preflight/i_track/receipt.json"
access_pin="$report/data/preflight/d1/access/receipt.json"
protocol="$report/D12_PROTOCOL.md"
cpu=(ChemBERTa_R11 FiLM_R12 Concat_R12)
gpu=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation)
workflow=(--models "${cpu[@]}" "${gpu[@]}" --access "$access" --access-reference "$access_pin"
          --original-plan "$old_plan" --original-reference "$old_pin" --cache "$r15/cache/GSE92742"
          --protocol "$protocol" --forecasts "$run/forecasts" --freeze "$run/freeze.json")

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$phase" != gpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

case "$phase" in
  stage|cpu|gpu)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^(0|[1-9][0-9]*)$ ]]
    roles=(A B C)
    slice="f$((task%9/3))-${roles[task%3]}"
    bundle="$run/inputs/$slice"
    if [[ "$phase" == stage ]]; then
      (( task < 9 ))
      exec "$python" -m src.r19_plan.d1_inputs --slice "$slice" --access "$access" --reference "$access_pin" \
        --approved-bundle "$old/predict-inputs/$slice" --approved-reference "$old_pin" --out "$bundle"
    fi
    if [[ "$phase" == cpu ]]; then
      (( task < 27 ))
      model="${cpu[task/9]}"
    else
      (( task < 36 ))
      model="${gpu[task/9]}"
    fi
    exec "$python" -m src.r19_plan.d12_predict --bundle "$bundle" --access-reference "$access_pin" \
      --slice "$slice" --model "$model" --protocol "$protocol" --config "$r12/configs/r12_multiaxis/config.yaml" \
      --assets "$r15/model-inputs" --prnet "$t2/t2-public/upstream-prnet" --out "$run/forecasts/$model/$slice"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.d12_workflow freeze "${workflow[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.d12_workflow score "${workflow[@]}" --vehicles "$r17/vehicles" \
      --expected-sha256 "$digest" --out "$run/score"
    ;;
  *) printf 'Unknown D1/D2 phase %s\n' "$phase" >&2; exit 2 ;;
esac
