#!/usr/bin/env bash
# Follow-up integrity fixtures (plan appendix C, items 1-4), CPU.
#SBATCH --job-name=r6f_tests
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
source experiments/r6_followup_v1/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID:-} commit=$(git rev-parse HEAD) start=$(date -Is)"
"$R6_PY" -m unittest -v tests.r6_followup_v1.test_followup
echo "end=$(date -Is)"
