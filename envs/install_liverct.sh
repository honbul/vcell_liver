#!/usr/bin/env bash
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
export CONDA_PKGS_DIRS=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/conda-pkgs
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
if [ ! -x .venvs/vcell-liverct/bin/python ]; then
  /usr/local/anaconda3/bin/conda create --prefix "$PWD/.venvs/vcell-liverct" --override-channels -c defaults -y python=3.9.7 pip
fi
P=.venvs/vcell-liverct/bin/python
$P -m pip install --upgrade pip==23.3.2 setuptools==59.5.0 'wheel<0.46' typing_extensions==4.12.2
$P -m pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$P -m pip install numpy==1.22.4 scipy==1.8.1 pandas==1.4.3 anndata==0.8.0 matplotlib==3.5.1 scanpy==1.9.1 scvi-tools==0.14.6 torchmetrics==0.7.3 pytorch-lightning==1.3.8 PyYAML==5.4.1 scikit-learn==1.1.1 joblib==1.1.0 xgboost==1.7.6 numba==0.56.4 h5py==3.8.0 statsmodels==0.13.5 seaborn==0.12.2 pyro-ppl==1.8.6 tensorboard==2.11.2 protobuf==3.20.3 psutil
$P -m pip install --no-deps scarches==0.5.9
$P -c 'import torch,scvi,scarches;print(torch.__version__,scvi.__version__,torch.cuda.is_available())'
$P -m pip freeze > envs/vcell-liverct.freeze.txt
