#!/usr/bin/env bash
#SBATCH --job-name=r1_checks
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/r1_checks_%j.log
# Solver and evaluation fixtures (must pass before real-data model fits).
set -euo pipefail
source experiments/softprior_multitask_r1/env.sh
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R1_PY" -m pytest -q -rs -p no:cacheprovider "$R1_SRC/tests" 2>&1 | tee reports/softprior_multitask_r1/tables/checks_pytest.txt
echo "end=$(date -Is)"
