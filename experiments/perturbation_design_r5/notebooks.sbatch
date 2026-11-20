#!/usr/bin/env bash
#SBATCH -J r5_nb
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 01:00:00
#SBATCH -o assets/logs/%x_%j.out
# Build and execute the tutorial notebook against the committed tables and the source store. No HCC treated value is read.
set -euo pipefail
source experiments/perturbation_design_r5/env.sh
export R5_NB_OUT="$R5_PUBLIC/tutorials/perturbation_design_r5"
mkdir -p "$R5_NB_OUT"
echo "start=$(date -Is) commit=$R5_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m ipykernel install --prefix "$R5_ASSETS/jupyter" --name python3 --display-name "Python 3" >/dev/null
export JUPYTER_PATH="$R5_ASSETS/jupyter/share/jupyter"
"$TB_PY" tutorials/perturbation_design_r5/build_notebooks.py
for nb in "$R5_NB_OUT"/*.ipynb; do
  "$TB_PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 "$nb"
done
echo "end=$(date -Is)"
