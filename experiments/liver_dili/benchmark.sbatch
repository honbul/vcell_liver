#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=02:00:00
set -euo pipefail

root=$1
mode=$2
phase=$3
base=$(dirname "$root")
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$root/deps:$root/embed-deps-v2:$root/embed-deps"
cd "${DILI_V1_CODE:-$root/code}"
common=(--splits "$root/splits-v1")
case "$phase" in
  predict)
    job=${SLURM_ARRAY_TASK_ID:?}
    if (( job < 5 )); then
      fold="murcko-$job"
    elif (( job < 10 )); then
      fold="butina-$(( job - 5 ))"
    else
      fold=release-0
    fi
    extra=()
    if [[ "$mode" != structure ]]; then
      extra=(--embeddings "$root/embeddings-full-v1/chemberta.parquet" --assays "$root/assays-v1")
    fi
    exec "$python" -m liver_dili.benchmark predict "${common[@]}" --fold "$fold" \
      --out "$root/$mode-forecasts/$fold" "${extra[@]}"
    ;;
  score)
    exec "$python" -m liver_dili.benchmark score "${common[@]}" \
      --forecasts "$root/$mode-forecasts" --out "$root/$mode-score"
    ;;
  *) exit 2 ;;
esac
