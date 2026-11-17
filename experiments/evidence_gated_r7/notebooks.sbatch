#!/usr/bin/env bash
#SBATCH -J r7_nb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 01:00:00
#SBATCH -o assets/logs/%x_%j.out
# Build and execute the tutorial notebook (frozen checkpoints on validation drugs + committed tables). Nothing is trained.
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
export R7_NB_OUT="$R7_PUBLIC/tutorials/evidence_gated_r7"
mkdir -p "$R7_NB_OUT"
echo "start=$(date -Is) commit=$R7_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" -m ipykernel install --prefix "$R7_ASSETS/jupyter" --name python3 --display-name "Python 3" >/dev/null
export JUPYTER_PATH="$R7_ASSETS/jupyter/share/jupyter"
"$TB_PY" tutorials/evidence_gated_r7/build_notebooks.py
for nb in "$R7_NB_OUT"/*.ipynb; do
  "$TB_PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 "$nb"
done
echo "end=$(date -Is)"
