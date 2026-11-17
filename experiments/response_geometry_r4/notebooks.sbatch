#!/usr/bin/env bash
#SBATCH -J r4_nb
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 01:00:00
#SBATCH -o assets/logs/%x_%j.out
# Build and execute the tutorial notebooks against committed tables (01) and the server store + one frozen checkpoint on
# validation drugs (02). Outputs go to the public mirror; copied into the repository with sync_public.sh. Nothing is trained.
set -euo pipefail
source experiments/response_geometry_r4/env.sh
export R4_NB_OUT="$R4_PUBLIC/tutorials/response_geometry_r4"
mkdir -p "$R4_NB_OUT"
echo "start=$(date -Is) commit=$R4_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m ipykernel install --prefix "$R4_ASSETS/jupyter" --name python3 --display-name "Python 3" >/dev/null
export JUPYTER_PATH="$R4_ASSETS/jupyter/share/jupyter"
"$TB_PY" tutorials/response_geometry_r4/build_notebooks.py
for nb in "$R4_NB_OUT"/*.ipynb; do
  "$TB_PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 "$nb"
done
echo "end=$(date -Is)"
