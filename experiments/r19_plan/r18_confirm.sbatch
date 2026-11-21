#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
access="$3"
index="$4"
structure="$5"
base="$6"
t2="$base/vcell_liver_goal_tracks/v1"
r15="$base/vcell_liver_r15_lincs/v1"
r12="$base/vcell_liver_r12_multiaxis"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
report="reports/r19_plan"
registered="$report/data/preflight/r18"
d0_reference="$report/data/r17-reconciliation.json"
protocol="$report/R18_CONFIRMATION_PROTOCOL.md"
landmarks="$r15/cache/GSE92742/landmarks.tsv"

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

indexed=(--index "$index" --index-reference "$registered/index" --landmarks "$landmarks")
score_args=(--forecasts "$run/forecasts" --access "$access" --access-ref "$registered/access/receipt.json"
            --protocol "$protocol" --d0-ref "$d0_reference" --freeze-file "$run/freeze.json")

case "$phase" in
  extract|cpu|gpu|merge)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    models=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation)
    if [[ "$phase" != "gpu" ]]; then
      limit=108
      export CUDA_VISIBLE_DEVICES=""
    else
      limit=432
    fi
    if [[ ! "$task" =~ ^(0|[1-9][0-9]*)$ ]] || (( task >= limit )); then
      printf '%s\n' "Invalid R18 array task $task for $phase" >&2
      exit 2
    fi
    slice="$("$python" -c 'import json, sys
from pathlib import Path
entries = sorted(json.loads(Path(sys.argv[1]).read_text())["files"])
if len(entries) != 108:
    raise ValueError("Expected the 108 registered R18 slices")
print(entries[int(sys.argv[2])])' "$registered/access/receipt.json" "$((task % 108))")"
    bundle="$run/predict-inputs/$slice"
    if [[ "$phase" = "extract" ]]; then
      exec "$python" -m src.r19_plan.r18_input_bundle --slice "$slice" --access "$access" \
        --access-reference "$registered/access/receipt.json" --out "$bundle" "${indexed[@]}"
    fi
    predictor=(--slice "$slice" --bundle "$bundle" --parts "$run/parts" --protocol "$protocol"
               --access-reference "$registered/access/receipt.json" --d0-reference "$d0_reference")
    if [[ "$phase" = "merge" ]]; then
      exec "$python" -m src.r19_plan.r18_predict merge "${predictor[@]}" --out "$run/forecasts"
    fi
    model=cpu
    if [[ "$phase" = "gpu" ]]; then
      model="${models[task/108]}"
    fi
    exec "$python" -m src.r19_plan.r18_predict predict "${predictor[@]}" --model "$model" \
      --config "$r12/configs/r12_multiaxis/config.yaml" --assets "$structure" \
      --asset-reference "$registered/structure/structure_receipt.json" --prnet "$t2/t2-public/upstream-prnet"
    ;;
  freeze)
    export CUDA_VISIBLE_DEVICES=""
    digest="$("$python" -m src.r19_plan.r18_score_contract freeze "${score_args[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    export CUDA_VISIBLE_DEVICES=""
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.r18_score "${score_args[@]}" "${indexed[@]}" \
      --expected-sha256 "$digest" --reports reports --out "$run/score"
    ;;
  *)
    printf '%s\n' "Unknown R18 stage $phase" >&2
    exit 2
    ;;
esac
