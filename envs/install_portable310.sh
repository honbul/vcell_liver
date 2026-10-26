#!/usr/bin/env bash
# A node-independent interpreter; existing scientific environment packages remain untouched.
set -euo pipefail
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
export CONDA_PKGS_DIRS=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/cache/conda-pkgs
if [ ! -x .venvs/vcell-runtime310/bin/python ]; then
  /usr/local/anaconda3/bin/conda create --prefix "$PWD/.venvs/vcell-runtime310" --override-channels -c defaults -y python=3.10.10 pip
fi
.venvs/vcell-runtime310/bin/python -c 'import sys;assert sys.version_info[:2]==(3,10);print(sys.version)'
/usr/local/anaconda3/bin/conda list --prefix "$PWD/.venvs/vcell-runtime310" --json > envs/vcell-runtime310.conda.json
