#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail

phase="$1"
run="$2"
base="$3"
x="$base/vcell_liver_r19_plan/x-a.KfInaHqK"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
pin=reports/r19_plan/data/x_a/access/receipt.json
protocol=reports/r19_plan/X3_PROTOCOL.md
workflow=(--access "$x/access" --pin "$pin" --protocol "$protocol"
          --forecasts "$run/forecasts" --freeze "$run/freeze.json")

export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

case "$phase" in
  stage|predict)
    task="${SLURM_ARRAY_TASK_ID:?expected an array task}"
    [[ "$task" =~ ^[0-8]$ ]]
    roles=(A B C)
    slice="HEPG2-f$((task/3))-${roles[task%3]}"
    if [[ "$phase" == stage ]]; then
      exec "$python" -m src.r19_plan.x3_workflow stage --access "$x/access" --pin "$pin" --slice "$slice" \
        --out "$run/donors/$slice"
    fi
    exec "$python" -m src.r19_plan.x3_workflow predict --pin "$pin" --slice "$slice" \
        --bundle "$x/inputs/$slice" --basal "$x/basal" --donors "$run/donors/$slice" \
        --protocol "$protocol" --out "$run/forecasts/$slice"
    ;;
  freeze)
    digest="$("$python" -m src.r19_plan.x3_workflow freeze "${workflow[@]}")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    (set -o noclobber; printf '%s\n' "$digest" > "$run/freeze.sha256")
    ;;
  score)
    IFS= read -r digest < "$run/freeze.sha256"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    exec "$python" -m src.r19_plan.x3_workflow score "${workflow[@]}" --expected-sha256 "$digest" --out "$run/score"
    ;;
  *) printf 'Unknown X3 phase %s\n' "$phase" >&2; exit 2 ;;
esac
