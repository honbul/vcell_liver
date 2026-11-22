#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
base="$3"
r15="$base/vcell_liver_r15_lincs/v1"
r17="$base/vcell_liver_r17_lincs/v1"
t2="$base/vcell_liver_goal_tracks/v1"
source="$base/vcell_liver_dev_i/efficacy.wFFiyzRF"
plan="$base/vcell_liver_dev_i/i-preflight.TE8L9Fqo/locked"
inst=/mnt/S1/sdata/pubdata/LINCS/GSE92742/GSE92742_Broad_LINCS_inst_info.txt
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
report=reports/r19_plan
common=(--split-reference "$report/data/preflight/i_track/receipt.json"
        --dose-reference "$report/data/x_l_support/l1/receipt.json" --protocol "$report/SL_PROTOCOL.md")
workflow=(--plan "$plan" --cache "$r15/cache/GSE92742" --forecasts "$run/forecasts"
          --freeze "$run/freeze.json" "${common[@]}")

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

case "$phase" in
  predict)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^[0-8]$ ]]
    roles=(A B C)
    slice="f$((task/3))-${roles[task%3]}"
    exec "$python" -m src.r19_plan.sl_predict --slice "$slice" --bundle "$source/predict-inputs/$slice" \
      --config "$base/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml" --assets "$r15/model-inputs" \
      --instances "$inst" --out "$run/forecasts/$slice" "${common[@]}"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.sl_workflow freeze "${workflow[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.sl_workflow score "${workflow[@]}" --vehicles "$r17/vehicles" \
      --instances "$inst" --expected-sha256 "$digest" --out "$run/score"
    ;;
  *) printf 'Unknown S/L phase %s\n' "$phase" >&2; exit 2 ;;
esac
