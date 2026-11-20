#!/usr/bin/env bash
#SBATCH --job-name=psa1_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/psa1_build_%j.log
# Stage A0/A1: record raw files, build registry, gene axis, cached inputs, split table; run checks.
set -euo pipefail
source experiments/pathway_sensitivity_a1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$PSA1_PY" -I "$PSA1_SRC/fetch_public.py" "$PWD/assets" "$PWD/manifests/pathway_sensitivity_a1/downloads.tsv"
"$PSA1_PY" "$PSA1_SRC/build_data.py" --rscript "$PSA1_RSCRIPT"
"$PSA1_PY" -m pytest -q -p no:cacheprovider "$PSA1_SRC/tests"
"$PSA1_PY" - <<'PY' > manifests/pathway_sensitivity_a1/environment.txt
import sys, sklearn, numpy, scipy, pandas
print("python", sys.version.split()[0]); print("scikit-learn", sklearn.__version__); print("numpy", numpy.__version__)
print("scipy", scipy.__version__); print("pandas", pandas.__version__)
PY
"$PSA1_RSCRIPT" -e 'cat("R", as.character(getRversion()), "\nsingscore", as.character(packageVersion("singscore")), "\n")' >> manifests/pathway_sensitivity_a1/environment.txt
echo "end=$(date -Is)"
