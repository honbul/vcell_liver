#!/bin/bash
#SBATCH --job-name=r10_env
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=assets/logs/%x_%j.out
# Own CPU-only Python env for R10 under assets/envs/r10 (caches inside assets/).
set -euo pipefail
ROOT=$(pwd)
export HOME=$ROOT/assets/home XDG_CACHE_HOME=$ROOT/assets/cache CONDA_PKGS_DIRS=$ROOT/assets/cache/conda_pkgs \
       PIP_CACHE_DIR=$ROOT/assets/cache/pip
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"
ENV=$ROOT/assets/envs/r10
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -q -p "$ENV" -c conda-forge --override-channels python=3.11 pip
fi
"$ENV/bin/python" -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
"$ENV/bin/python" -m pip install -q numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 scikit-learn==1.5.1 \
  matplotlib==3.9.1 pyarrow==16.1.0 pot==0.9.4
"$ENV/bin/python" -I -c "import numpy, scipy, pandas, sklearn, torch, ot, matplotlib; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__, torch.__version__, ot.__version__)"
"$ENV/bin/python" -m pip freeze > "$ROOT/assets/envs/r10_pip_freeze.txt"
conda list -p "$ENV" > "$ROOT/assets/envs/r10_conda_list.txt"
