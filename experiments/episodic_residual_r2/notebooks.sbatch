#!/usr/bin/env bash
#SBATCH -J r2_nb
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 01:00:00
#SBATCH -o assets/logs/%x_%j.out
# Build and execute the tutorial notebooks against the committed tables. Outputs go to the public mirror (the server
# worktree stays clean); they are copied into the repository with sync_public.sh. Nothing is trained, no HCC array is read.
set -euo pipefail
source experiments/episodic_residual_r2/env.sh
export R2_NB_OUT="$R2_PUBLIC/tutorials/episodic_residual_r2"
mkdir -p "$R2_NB_OUT"
echo "start=$(date -Is) commit=$R2_COMMIT host=$(hostname) job=${SLURM_JOB_ID:-}"
"$TB_PY" tutorials/episodic_residual_r2/build_notebooks.py
for nb in "$R2_NB_OUT"/*.ipynb; do
  "$TB_PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1800 "$nb"
done
echo "end=$(date -Is)"
