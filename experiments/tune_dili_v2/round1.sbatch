#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
workspace=$2
run=$3
shift 3
cd "$workspace"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|1|2|3|4) kind=murcko; fold=$SLURM_ARRAY_TASK_ID ;;
  *) kind=butina; fold=$((SLURM_ARRAY_TASK_ID - 5)) ;;
esac
export PYTHONPATH="$workspace:$study/liver_dili_v1/deps:$study/vc_build_v1/v0plus-model-deps"
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m liver_dili_v2.round1 --data "$study/liver_dili_v1/splits-v1" \
  --fm "$study/nextgen_5track/fm_store_v1/chemistry-v1" \
  --chemberta "$study/liver_dili_v1/embeddings-full-v1/chemberta.parquet" \
  --kind "$kind" --fold "$fold" --out "$run/$kind-$fold" "$@"
