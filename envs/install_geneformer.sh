#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
/usr/local/anaconda3/bin/python -m venv .venvs/vcell-geneformer
P=.venvs/vcell-geneformer/bin/python
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
$P -m pip install --upgrade 'pip<26' wheel setuptools
$P -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$P -m pip install numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 anndata==0.10.9 scanpy==1.10.4 scikit-learn==1.5.2 datasets==2.21.0 pyarrow==22.0.0 numba==0.60.0 matplotlib==3.9.4 seaborn==0.13.2 umap-learn==0.5.7 transformers==4.46.0 peft==0.13.2 accelerate==1.2.1 bitsandbytes==0.45.5 loompy==3.0.8 optuna==4.2.1 optuna-integration==4.2.1 ray==2.40.0 tdigest==0.5.2.2 tensorboard==2.18.0 pyyaml psutil nbformat nbclient ipykernel ipywidgets
$P -m pip install --no-deps -e third_party/Geneformer
$P -c 'from geneformer import TranscriptomeTokenizer; import torch; print(torch.__version__, torch.cuda.is_available())'
$P -m pip freeze > envs/vcell-geneformer.freeze.txt
