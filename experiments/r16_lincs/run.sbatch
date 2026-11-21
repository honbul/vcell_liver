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

protocol="reports/r16_lincs/PROTOCOL.md"
r15_protocol="reports/r15_lincs/PROTOCOL.md"
out="$run/internal${R16_OUTPUT_SUFFIX:-}"
cache="$prior/cache"
assets="$prior/model-inputs"
extra=()
if [[ "${4:-}" != "" ]]; then
    protocol="reports/r16_lincs/CONFIRMATION.md"
    out="$run/confirm${R16_OUTPUT_SUFFIX:-}/$4"
    cache="$run/confirmation-cache"
    assets="$run/confirmation-assets"
    extra=(--confirm --cell "$4")
fi
shared=(--cache "$cache" --protocol "$protocol" --r15-protocol "$r15_protocol" --prior "$prior" --out "$out")
case "$stage" in
  predict)
    task="$SLURM_ARRAY_TASK_ID"
    roles=(A B)
    case "$3" in
      proposed) models=(R15 blend_FiLM blend_concat) ;;
      legacy_cpu) models=(ChemBERTa_R11 FiLM_R12 Concat_R12) ;;
      legacy_gpu|legacy_gpu_missing) models=(chemCPA_P0 TranSiGen_Tahoe XPert_morgan PRnet-well-adaptation) ;;
      *) exit 2 ;;
    esac
    if [[ "$3" = legacy_gpu_missing ]]; then
        scenario="${roles[task%12/6]}"
        fold="$((task%6/2))"
        role="${roles[task%2]}"
        if [[ -s "$out/predict/$scenario/f$fold-$role/${models[task/12]}/receipt.json" ]]; then
            exit 0
        fi
    fi
    exec "$python" -m src.r16_lincs.predict "${shared[@]}" "${extra[@]}" --assets "$assets" \
      --config "$r12/configs/r12_multiaxis/config.yaml" --prnet "$t2/t2-public/upstream-prnet" \
      --model "${models[task/12]}" --scenario "${roles[task%12/6]}" \
      --fold "$((task%6/2))" --role "${roles[task%2]}"
    ;;
  score)
    exec "$python" -m src.r16_lincs.score "$3" "${shared[@]}" "${extra[@]}"
    ;;
  audit)
    exec "$python" -m src.r16_lincs.audit --base "$base" --prior "$prior" --out "$run/audit"
    ;;
  stage)
    exec "$python" -m src.r16_lincs.audit --base "$base" --prior "$prior" --out "$run/confirmation-cache" \
      --audit "$run/audit" --protocol "$protocol"
    ;;
  features)
    exec "$python" -m src.r16_lincs.confirmation --cache "$run/confirmation-cache" \
      --out "$run/confirmation-assets" --checkpoint "$prior/chemberta" \
      --reference reports/r11_chemical_probe/manifest/feature_receipt.json \
      --gene-map "$base/vcell_liver_transigen_adaptation_b1/assets/processed/cmap2020_v1/genes.tsv"
    ;;
  *) exit 2 ;;
esac
