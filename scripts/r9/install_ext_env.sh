#!/bin/bash
# External comparison env (scPoli via scArches from PyPI; ProtoCloud from its public GitHub at a pinned commit).
set -euo pipefail
R9_ROOT=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state
cd "$R9_ROOT"
export CONDA_PKGS_DIRS="$R9_ROOT/assets/cache/conda-pkgs" PIP_CACHE_DIR="$R9_ROOT/assets/cache/pip"
export HOME="$R9_ROOT/assets/home" XDG_CACHE_HOME="$R9_ROOT/assets/cache/xdg"
ENV="$R9_ROOT/assets/envs/r9ext"
PROTOCLOUD_COMMIT=90c2678f9ce54e1e9911bd1aa5086c63a7798d89
if [ ! -x "$ENV/bin/python" ]; then
  /usr/local/anaconda3/bin/conda create --prefix "$ENV" --override-channels -c conda-forge -y python=3.10.14 pip
fi
"$ENV/bin/python" -m pip install --no-input torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/python" -m pip install --no-input scarches==0.6.1 scanpy==1.10.4 anndata==0.10.9 numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2
git -C assets/third_party/ProtoCloud checkout -q "$PROTOCLOUD_COMMIT"
"$ENV/bin/python" -m pip install --no-input --no-deps assets/third_party/ProtoCloud
"$ENV/bin/python" -m pip install --no-input matplotlib-venn seaborn umap-learn
"$ENV/bin/python" -c 'import scarches, ProtoCloud, torch, scanpy; print("scarches", scarches.__version__, "torch", torch.__version__, "ProtoCloud ok")'
"$ENV/bin/python" -m pip freeze > "$R9_ROOT/manifests/r9_rare_state/env_r9ext.freeze.txt"
