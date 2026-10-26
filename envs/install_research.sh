#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
/usr/local/anaconda3/bin/python -m venv .venvs/vcell-research
P=.venvs/vcell-research/bin/python
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
$P -m pip install --upgrade pip==25.3 wheel setuptools==75.8.0
$P -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$P -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 anndata==0.11.4 zarr==2.18.3 numcodecs==0.12.1 scanpy==1.10.4 scikit-learn==1.5.2 numba==0.60.0 matplotlib==3.9.4 umap-learn==0.5.7 scikit-misc==0.5.1 pooch==1.8.2 pyyaml psutil nbformat nbclient ipykernel local-attention==1.9.15 einops==0.8.0 scvi-tools==1.2.2 jax==0.4.35 jaxlib==0.4.35 flax==0.10.2 numpyro==0.15.3 optax==0.2.4 chex==0.1.90 orbax-checkpoint==0.10.3 lightning==2.4.0 torchmetrics==1.5.2 torch-geometric==2.6.1 networkx==3.3 dcor==0.6 gdown==5.2.0
$P -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
$P -m pip install --no-deps -e third_party/GEARS
$P -m pip freeze > envs/vcell-research.freeze.txt
$P -c 'import torch,scvi,gears;print(torch.__version__,scvi.__version__,torch.cuda.is_available())'
