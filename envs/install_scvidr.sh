#!/usr/bin/env bash
# Child environment reads the LiverCT stack but shadows scvi with the native VIDR API.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
export PIP_CACHE_DIR=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/pip
if [ ! -x .venvs/vcell-scvidr/bin/python ]; then
  .venvs/vcell-liverct/bin/python -m venv --system-site-packages .venvs/vcell-scvidr
fi
.venvs/vcell-scvidr/bin/python -m pip install --no-deps adjustText==0.7.3 scvi-tools==0.13.0
.venvs/vcell-scvidr/bin/python -c 'import sys;sys.path.insert(0,"third_party/scVIDR/vidr");from vidr import VIDR;import torch,scvi;print(torch.__version__,scvi.__version__)'
.venvs/vcell-scvidr/bin/python -m pip freeze > envs/vcell-scvidr.freeze.txt
