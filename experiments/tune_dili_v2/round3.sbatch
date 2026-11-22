#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=01:30:00
set -euo pipefail
study=$1
code=$2
artifacts=$3
out=$4
cd "$code"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|1|2|3|4) kind=murcko; fold=$SLURM_ARRAY_TASK_ID ;;
  *) kind=butina; fold=$((SLURM_ARRAY_TASK_ID - 5)) ;;
esac
export PYTHONPATH="$code:$study/liver_dili_v1/deps:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m liver_dili_v2.round3 --data "$study/liver_dili_v1/splits-v1" \
  --fm "$study/nextgen_5track/fm_store_v1/chemistry-v1" \
  --chemberta "$study/liver_dili_v1/embeddings-full-v1/chemberta.parquet" \
  --round1-cpu "$artifacts/round1-03-cpu" --round1-cat "$artifacts/round1-03-cat" \
  --round1-kernels "$artifacts/round1-04-kernels" --round2 "$artifacts/round2-01" --kind "$kind" --fold "$fold" --out "$out/$kind-$fold"
