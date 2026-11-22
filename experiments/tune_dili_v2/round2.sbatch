#!/usr/bin/env bash
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=02:00:00
set -euo pipefail
study=$1
workspace=$2
run=$3
mechanisms=$4
cd "$workspace"
case "${SLURM_ARRAY_TASK_ID:?}" in
  0|1|2|3|4) kind=murcko; fold=$SLURM_ARRAY_TASK_ID ;;
  *) kind=butina; fold=$((SLURM_ARRAY_TASK_ID - 5)) ;;
esac
runtime="$study/vcell_liver/vcell_r12_runtime_cu126"
export PYTHONPATH="$workspace:$runtime:$study/liver_dili_v1/deps"
export OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
python="$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
"$python" -m liver_dili_v2.round2 --data "$study/liver_dili_v1/splits-v1" \
  --aux "$mechanisms" --all-fda-structures "$study/liver_dili_v1/chemistry-v1/fda_structures.tsv" \
  --kind "$kind" --fold "$fold" --out "$run/$kind-$fold"
