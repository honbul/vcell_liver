#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
P=.venvs/vcell-scgpt/bin/python
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
$P -m pip install --upgrade 'pip<26' wheel setuptools
$P -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# Inference-only dependencies: upstream scvi/GEARS training extras are deliberately excluded.
$P -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 anndata==0.10.9 scanpy==1.10.4 scikit-learn==1.5.2 datasets==2.21.0 pyarrow==22.0.0 numba==0.60.0 matplotlib==3.9.4 seaborn==0.13.2 umap-learn==0.5.7 scikit-misc==0.5.1 gdown==5.2.0 pyyaml psutil nbformat nbclient ipykernel ipywidgets huggingface-hub==0.36.0 wandb==0.19.11
$P -m pip install --no-deps -e third_party/scGPT
$P -c 'import scgpt, torch; print(scgpt.__version__, torch.__version__, torch.cuda.is_available())'
$P -m pip freeze > envs/vcell-scgpt.freeze.txt
