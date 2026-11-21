#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0 R13_DEVICE=cpu
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
r12="$PWD/../vcell_liver_r12_multiaxis/assets"
asset="$PWD/assets/external-gse306429"
export PYTHONPATH="$asset/python310-h5py-3.15.1:${PYTHONPATH:-}"
run="$PWD/runs/$2"
common=(--config configs/r12_multiaxis/config.yaml --assets "$r12" --manifest "$asset/manifest-v3"
  --h5 "$asset/GSE306429_combined_pseudobulk.h5ad")
case "$1" in
  plan)
    exec "$python" -m src.r13_optimization.external plan "${common[@]}" --out "$run/cohort"
    ;;
  predict)
    "$python" -m src.r13_optimization.external_tune lock --manifest "$asset/manifest-v3" \
      --runs "$PWD/runs/$3" --recipe "$run/recipe.json"
    exec "$python" -m src.r13_optimization.external predict "${common[@]}" \
      --recipe "$run/recipe.json" --out "$run/frozen"
    ;;
  score)
    exec "$python" -m src.r13_optimization.external score "${common[@]}" \
      --recipe "$run/recipe.json" --out "$run/frozen" --scores "$run/scored"
    ;;
  *) exit 2 ;;
esac
