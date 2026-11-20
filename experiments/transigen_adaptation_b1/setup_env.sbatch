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
if [ ! -x "$ENV/bin/python" ]; then python3 -m venv --copies "$ENV"; fi
"$ENV/bin/python" -m pip install -q --no-cache-dir --disable-pip-version-check torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/python" -m pip install -q --no-cache-dir --disable-pip-version-check numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 \
    scikit-learn==1.5.1 h5py==3.11.0 rdkit==2024.3.5 matplotlib==3.9.2 pyyaml==6.0.2 nbconvert nbformat ipykernel
"$ENV/bin/python" - <<'PY'
import torch, rdkit, h5py, numpy, pandas, sklearn, scipy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| rdkit", rdkit.__version__, "| h5py", h5py.__version__,
      "| numpy", numpy.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__, "| scipy", scipy.__version__)
PY
"$ENV/bin/pip" freeze > manifests/transigen_adaptation_b1/env_pip_freeze.txt
echo "end=$(date -Is)"
