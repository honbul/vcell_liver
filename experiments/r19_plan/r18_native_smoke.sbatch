#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail

phase=$1
run=$2
base=$(dirname "$(dirname "$run")")
repo="$run/repo"
r15="$base/vcell_liver_r15_lincs/v1"
r12="$base/vcell_liver_r12_multiaxis"
t2="$base/vcell_liver_goal_tracks/v1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
protocol="$repo/reports/r19_plan/R18_CONFIRMATION_PROTOCOL.md"
d0="$repo/reports/r19_plan/data/r17-reconciliation.json"

export PYTHONPATH="$repo:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$t2/t2/dependencies:$t2/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$phase" != gpu ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

fixture="$run/fixture"
case "$phase" in
  build)
    exec /usr/bin/time -v "$python" -u -m src.r19_plan.r18_native_smoke build --out "$fixture" \
      --landmarks "$r15/cache/GSE92742/landmarks.tsv" --gene-map "$r15/model-inputs/metadata/genes.tsv" \
      --d0-reference "$d0"
    ;;
  verify)
    exec /usr/bin/time -v "$python" -u -m src.r19_plan.r18_native_smoke verify --fixture "$fixture" \
      --forecasts "$run/forecasts" --protocol "$protocol" --d0-reference "$d0" --out "$run/verified.json"
    ;;
  cpu|gpu|merge)
    cases=(C2/B/f1-B C3_24h_HEPG2_only/A/f2-C)
    task=${SLURM_ARRAY_TASK_ID:?}
    slice="${cases[task % 2]}"
    args=(--slice "$slice" --bundle "$fixture/bundles/$slice" --parts "$run/parts"
          --protocol "$protocol" --access-reference "$fixture/access.json" --d0-reference "$d0")
    if [[ "$phase" == merge ]]; then
      exec /usr/bin/time -v "$python" -u -m src.r19_plan.r18_predict merge "${args[@]}" --out "$run/forecasts"
    fi
    model=cpu
    if [[ "$phase" == gpu ]]; then
      models=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation)
      model=${models[task / 2]}
    fi
    exec /usr/bin/time -v "$python" -u -m src.r19_plan.r18_predict predict "${args[@]}" --model "$model" \
      --config "$r12/configs/r12_multiaxis/config.yaml" --assets "$fixture/structure" \
      --asset-reference "$fixture/structure/receipt.json" --prnet "$t2/t2-public/upstream-prnet"
    ;;
esac
exit 2
