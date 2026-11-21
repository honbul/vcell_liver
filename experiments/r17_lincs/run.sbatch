#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
run="$1"
stage="$2"
base="$(dirname "$(dirname "$run")")"
prior="$base/vcell_liver_r15_lincs/v1"
t2="$base/vcell_liver_goal_tracks/v1"
r12="$base/vcell_liver_r12_multiaxis"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

protocol="reports/r17_lincs/PROTOCOL.md"
shared=(--cache "$prior/cache" --vehicles "$run/vehicles" --protocol "$protocol" \
        --r15-protocol reports/r15_lincs/PROTOCOL.md --out "$run")
case "$stage" in
  stage)
    exec "$python" -m src.r17_lincs.vehicles stage --cache "$prior/cache" --lincs /mnt/S1/sdata/pubdata/LINCS \
      --out "$run/vehicles" --audit "$run/support" --protocol "$protocol"
    ;;
  predict)
    task="$SLURM_ARRAY_TASK_ID"
    scenarios=(A B)
    roles=(A B C)
    if [[ "$3" = cpu ]]; then
        models=(cpu)
    else
        models=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation)
    fi
    exec "$python" -m src.r17_lincs.predict "${shared[@]}" --assets "$prior/model-inputs" \
      --config "$r12/configs/r12_multiaxis/config.yaml" --prnet "$t2/t2-public/upstream-prnet" \
      --model "${models[task/18]}" --scenario "${scenarios[task%18/9]}" \
      --fold "$((task%9/3))" --role "${roles[task%3]}"
    ;;
  score)
    exec "$python" -m src.r17_lincs.score "$3" "${shared[@]}"
    ;;
  inventory)
    exec "$python" -m src.r17_lincs.inventory --base "$base" --out "$run/inventory"
    ;;
  *) exit 2 ;;
esac
