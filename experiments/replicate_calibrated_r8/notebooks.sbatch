#!/usr/bin/env bash
#SBATCH -J r8_nb
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 01:00:00
#SBATCH -o assets/logs/%x_%j.out
# Build and execute the tutorial notebooks against the committed tables. Outputs go to the public mirror (the server
# worktree stays clean); they are copied into the repository with sync_public.sh. Nothing is trained, no HCC array is read.
set -euo pipefail
source experiments/replicate_calibrated_r8/env.sh
export R8_NB_OUT="$R8_PUBLIC/tutorials/replicate_calibrated_r8"
mkdir -p "$R8_NB_OUT"
echo "start=$(date -Is) commit=$R8_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
# kernel spec of the (read-only) study-B environment, registered under this study's own assets so the notebooks run with that interpreter
"$TB_PY" -m ipykernel install --prefix "$R8_ASSETS/jupyter" --name python3 --display-name "Python 3" >/dev/null
export JUPYTER_PATH="$R8_ASSETS/jupyter/share/jupyter"
"$TB_PY" tutorials/replicate_calibrated_r8/build_notebook.py
for nb in "$R8_NB_OUT"/*.ipynb; do
  "$TB_PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 "$nb"
done
echo "end=$(date -Is)"
