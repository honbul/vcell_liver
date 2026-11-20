#!/usr/bin/env bash
# Execute the R6 tutorial notebook in place (reads committed tables only; no GPU, no cell data).
# NB_PY must point at a python with pandas, matplotlib and nbconvert (an existing env is reused read-only).
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/depth_robust_r6/env.sh
export JUPYTER_CONFIG_DIR="$R6_ASSETS/home/jupyter" JUPYTER_DATA_DIR="$R6_ASSETS/home/jupyter_data" JUPYTER_RUNTIME_DIR="$R6_ASSETS/home/jupyter_rt" IPYTHONDIR="$R6_ASSETS/home/ipython" MPLCONFIGDIR="$R6_ASSETS/home/mpl"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$MPLCONFIGDIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"${NB_PY:?set NB_PY}" -m nbconvert --to notebook --execute --inplace tutorials/depth_robust_r6/01_thinning_and_results.ipynb
echo "end=$(date -Is)"
