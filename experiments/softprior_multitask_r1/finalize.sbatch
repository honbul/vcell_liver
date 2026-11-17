#!/usr/bin/env bash
#SBATCH --job-name=r1_fin
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r1_fin_%j.log
# After the frozen evaluation: figures from the committed tables, tutorial notebooks executed with
# outputs embedded, environment record. Nothing here fits or scores a study model.
set -euo pipefail
source experiments/softprior_multitask_r1/env.sh
export JUPYTER_CONFIG_DIR="$PWD/assets/external/jupyter/config" JUPYTER_DATA_DIR="$PWD/assets/external/jupyter/data"
export JUPYTER_RUNTIME_DIR="$PWD/assets/external/jupyter/runtime" IPYTHONDIR="$PWD/assets/external/jupyter/ipython"
mkdir -p "$JUPYTER_RUNTIME_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_CONFIG_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR"
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R1_PY" "$R1_SRC/make_figures.py" --tables reports/softprior_multitask_r1/tables --out reports/softprior_multitask_r1/figures
"$R1_PY" tutorials/softprior_multitask_r1/_build_notebooks.py
for nb in tutorials/softprior_multitask_r1/0*.ipynb; do
  "$R1_PY" -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb"
done
{ echo "# python: $("$R1_PY" --version 2>&1)"; echo "# octave: $(octave --version 2>/dev/null | head -1 || echo not-on-this-node)"; "$R1_PY" -m pip freeze 2>/dev/null; } > manifests/softprior_multitask_r1/environment.txt
echo "end=$(date -Is)"
