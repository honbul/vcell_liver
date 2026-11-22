#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
base="$3"
prior="$base/vcell_liver_r15_lincs/v1"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
report=reports/r19_plan
pin="$report/data/x_a/access/receipt.json"
protocol="$report/X_PROTOCOL.md"
workflow=(--access "$run/access" --pin "$pin" --protocol "$protocol"
          --forecasts "$run/forecasts" --freeze "$run/freeze.json")

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

case "$phase" in
  predict)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^(0|[1-9][0-9]*)$ ]] && (( task < 45 ))
    cells=(HEPG2 A375 A549 HT29 MCF7)
    roles=(A B C)
    slice="${cells[task/9]}-f$((task%9/3))-${roles[task%3]}"
    exec "$python" -m src.r19_plan.x_predict --slice "$slice" --bundle "$run/inputs/$slice" --pin "$pin" \
      --basal "$run/basal" --protocol "$protocol" --assets "$prior/model-inputs" \
      --config "$base/vcell_liver_r12_multiaxis/configs/r12_multiaxis/config.yaml" --out "$run/forecasts/$slice"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.x_workflow freeze "${workflow[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.x_workflow score "${workflow[@]}" --expected-sha256 "$digest" --out "$run/score"
    ;;
  *) printf 'Unknown X phase %s\n' "$phase" >&2; exit 2 ;;
esac
