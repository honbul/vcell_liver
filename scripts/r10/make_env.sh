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
       PIP_CACHE_DIR=$ROOT/assets/cache/pip MAMBA_ROOT_PREFIX=$ROOT/assets/cache/mamba
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"
ENV=$ROOT/assets/envs/r10
[ -x "$ENV/bin/python" ] || mamba create -y -p "$ENV" -c conda-forge --override-channels \
  python=3.11 numpy=1.26 scipy pandas=2.2 scikit-learn matplotlib pyarrow pot "pytorch=2.3=*cpu*"
"$ENV/bin/python" -I -c "import numpy, scipy, pandas, sklearn, torch, ot, matplotlib; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__, torch.__version__, ot.__version__)"
mamba list -p "$ENV" --explicit > "$ROOT/assets/envs/r10_explicit.txt"
mamba list -p "$ENV" > "$ROOT/assets/envs/r10_list.txt"
