#!/usr/bin/env bash
#SBATCH --job-name=r1_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/r1_build_%j.log
# Stage R1-0: fixed-ID gene axis, cached inputs, solver fixtures.
set -euo pipefail
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R1_PY" "$R1_SRC/build_data.py" --rscript "$R1_RSCRIPT"
"$R1_PY" -m pytest -q -p no:cacheprovider "$R1_SRC/tests" 2>&1 | tee reports/softprior_multitask_r1/tables/checks_pytest.txt
"$R1_PY" - <<'PY' > manifests/softprior_multitask_r1/environment.txt
import sys, sklearn, numpy, scipy, pandas
print("python", sys.version.split()[0]); print("scikit-learn", sklearn.__version__); print("numpy", numpy.__version__)
print("scipy", scipy.__version__); print("pandas", pandas.__version__)
try:
    import torch; print("torch", torch.__version__)
except Exception as e:
    print("torch", "not installed")
PY
echo "end=$(date -Is)"
