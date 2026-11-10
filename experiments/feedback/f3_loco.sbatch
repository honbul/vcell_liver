#!/usr/bin/env bash
# F3 step 1 finalisation (data contract) + step 2b/3 (deltas, DEG truth, LOCO baselines)
#SBATCH --job-name=f3_loco
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/f3_loco_%j.log
set -euo pipefail
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONHASHSEED=0
echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
cd src/feedback
../../.venvs/scgpt/bin/python -u f3_contract.py --pseudobulk ../../assets/f3/pseudobulk --out ../../reports/feedback/F3
cd ../..
.venvs/scgpt/bin/python -u src/feedback/f3_loco.py --pseudobulk assets/f3/pseudobulk --assets assets/f3 --out reports/feedback/F3 "$@"
.venvs/scgpt/bin/pip freeze > reports/feedback/F3/env_freeze.txt
echo "end=$(date -Is)"
