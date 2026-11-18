#!/bin/bash
# Build the node-independent R9 core env under assets/envs/r9core (git-ignored). Run via Slurm.
set -euo pipefail
R9_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state
cd "$R9_ROOT"
export CONDA_PKGS_DIRS="$R9_ROOT/assets/cache/conda-pkgs" PIP_CACHE_DIR="$R9_ROOT/assets/cache/pip"
export HOME="$R9_ROOT/assets/home" XDG_CACHE_HOME="$R9_ROOT/assets/cache/xdg"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HOME"
ENV="$R9_ROOT/assets/envs/r9core"
if [ ! -x "$ENV/bin/python" ]; then
  /usr/local/anaconda3/bin/conda create --prefix "$ENV" --override-channels -c conda-forge -y python=3.10.14 pip
fi
"$ENV/bin/python" -m pip install --no-input \
  numpy==1.26.4 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 anndata==0.10.9 h5py==3.12.1 \
  matplotlib==3.9.4 pyyaml==6.0.2 threadpoolctl==3.5.0
"$ENV/bin/python" -m pip install --no-input torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$ENV/bin/python" -c 'import sys,numpy,scipy,pandas,sklearn,anndata,matplotlib,torch;print(sys.version);print(numpy.__version__,scipy.__version__,pandas.__version__,sklearn.__version__,anndata.__version__,matplotlib.__version__,torch.__version__)'
"$ENV/bin/python" -m pip freeze > "$R9_ROOT/manifests/r9_rare_state/env_r9core.freeze.txt"
