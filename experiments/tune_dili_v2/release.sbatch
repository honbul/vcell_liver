#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=03:00:00
set -euo pipefail
study=$1
code=$2
runs=$3
stage=$4
out=$5
cd "$code"
runtime="$study/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$code:$runtime:$study/liver_dili_v1/deps:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
policy="$runs/round4-01/policy.json"
data="$study/liver_dili_v1/splits-v1"
fm="$study/nextgen_5track/fm_store_v1/chemistry-v1"
cb="$study/liver_dili_v1/embeddings-full-v1/chemberta.parquet"
base=(--data "$data" --policy "$policy" --fm "$fm" --chemberta "$cb")
case "$stage" in
  registered_folds)
    case "${SLURM_ARRAY_TASK_ID:?}" in
      0|1|2|3|4) kind=murcko; fold=$SLURM_ARRAY_TASK_ID ;;
      *) kind=butina; fold=$((SLURM_ARRAY_TASK_ID - 5)) ;;
    esac
    scope="$kind-$fold"
    "$python" -m liver_dili_v2.release fit_fold "${base[@]}" --runs "$runs" --kind "$kind" --fold "$fold" \
      --out "$out/models/$scope"
    "$python" -m liver_dili_v2.release predict "${base[@]}" --kind "$kind" --fold "$fold" \
      --checkpoint "$out/models/$scope" --out "$out/forecasts/$scope"
    ;;
  release_history_fit)
    "$python" -m liver_dili_v2.release fit_history "${base[@]}" --runs "$runs" --device cuda \
      --aux "$study/top3_tuning_v2/b/input/assays-04" --fda-structures "$study/liver_dili_v1/chemistry-v1/fda_structures.tsv" \
      --out "$out/models/release-0"
    ;;
  release_history_predict)
    "$python" -m liver_dili_v2.release predict "${base[@]}" --kind release --fold 0 \
      --checkpoint "$out/models/release-0" --out "$out/forecasts/release-0"
    ;;
  freeze)
    "$python" -m liver_dili_v2.release freeze "${base[@]}" --runs "$out/forecasts" --out "$out/freeze.json"
    ;;
  score)
    "$python" -m liver_dili_v2.release score "${base[@]}" --freeze "$out/freeze.json" --runs "$out/forecasts" \
      --baselines "$study/liver_dili_v1/full-v2-forecasts" --out "$out/score.json"
    ;;
  offline)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$python" -m liver_dili_v2.predict 'CCO' --out "$out/offline-smoke.json"
    ;;
  *) exit 2 ;;
esac
