#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
run="$1"
stage="$2"
base="$(dirname "$(dirname "$run")")"
prior="$base/vcell_liver_goal_tracks/v1"
r11="$base/vcell_liver_r11_chemical_probe/assets"
r12="$base/vcell_liver_r12_multiaxis"
b1="$base/vcell_liver_transigen_adaptation_b1/assets/processed/cmap2020_v1"
python="$r11/study-env/bin/python"
export PYTHONPATH="$PWD:$base/vcell_liver/vcell_r12_runtime_cu126:$base/vcell_liver/vcell_r12_runtime_transformers:$prior/t2/dependencies:$prior/t2-public/deps"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
protocol="reports/r15_lincs/PROTOCOL.md"
test "$(sha256sum "$protocol" | cut -d' ' -f1)" = ad8a19393c514d5b45cfa0bc98d34a3f871cad01e352c2f6df9113a42fd9a245
shared=(--cache "$run/cache" --protocol "$protocol" --assets "$run/model-inputs" --out "$run")
case "$stage" in
  cache)
    exec "$python" -m src.r15_lincs.cache --root /mnt/S1/sdata/pubdata/LINCS --audit "$run/audit" \
      --protocol "$protocol" --tahoe-entities "$r11/features/entities.tsv" --tahoe-split "$prior/t1-cache/split.json" \
      --out "$run/cache" --p23 "$base/vcell_liver_p123/v1/lincs" --study "$3"
    ;;
  features)
    exec "$python" -m src.r15_lincs.features --cache "$run/cache" --out "$run/model-inputs" \
      --lincs-map "$b1/genes.tsv" --reference reports/r11_chemical_probe/manifest/feature_receipt.json \
      --checkpoint "$run/chemberta"
    ;;
  predict)
    task="$SLURM_ARRAY_TASK_ID"
    roles=(A B)
    case "$3" in
      proposed) models=(R15) ;;
      legacy_cpu) models=(ChemBERTa_R11 FiLM_R12 Concat_R12) ;;
      legacy_gpu) models=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation) ;;
      *) exit 2 ;;
    esac
    exec "$python" -m src.r15_lincs.forecast "${shared[@]}" --config "$r12/configs/r12_multiaxis/config.yaml" \
      --prnet "$prior/t2-public/upstream-prnet" --model "${models[task/12]}" --scenario "${roles[task%12/6]}" \
      --fold "$((task%6/2))" --role "${roles[task%2]}"
    ;;
  internal)
    exec "$python" -m src.r15_lincs.score "$3" "${shared[@]}"
    ;;
  external)
    action="$3"
    opts=()
    if [[ "$action" = predict ]]; then
        task="$SLURM_ARRAY_TASK_ID"
        roles=(A B)
        opts=(--scenario "${roles[task/2]}" --role "${roles[task%2]}")
    fi
    exec "$python" -m src.r15_lincs.external "$action" "${shared[@]}" \
      --tahoe "$prior" --tahoe-assets "$r11" --t2-reference reports/goal_tracks/data/known_vc/d5/freeze.json \
      "${opts[@]}"
    ;;
  *) exit 2 ;;
esac
