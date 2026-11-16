#!/usr/bin/env bash
#SBATCH -J tb_env
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 02:00:00
#SBATCH -o logs/tb_env_%j.out
# Private environment for Track B. Does not touch any existing environment.
set -euo pipefail
source experiments/transigen_adaptation_b1/env.sh
ENV="$TB_ASSETS/envs/tsg"
if [ ! -x "$ENV/bin/python" ]; then
  /usr/local/bin/mamba create -y -q -p "$ENV" -c conda-forge python=3.10 pip numpy=1.26 pandas scipy scikit-learn h5py \
      rdkit matplotlib pyyaml nbconvert nbformat ipykernel
fi
"$ENV/bin/pip" install -q --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/python" - <<'PY'
import torch, rdkit, h5py, numpy, pandas, sklearn, scipy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| rdkit", rdkit.__version__, "| h5py", h5py.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__, "| scipy", scipy.__version__)
PY
"$ENV/bin/pip" freeze > manifests/transigen_adaptation_b1/env_pip_freeze.txt
echo "end=$(date -Is)"
