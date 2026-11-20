#!/usr/bin/env bash
# G4: fetch State ST-HVG-Replogle zeroshot assets (~12.4 GB) at a pinned revision.
#SBATCH --job-name=g4_state_fetch
#SBATCH --partition=cpu
#SBATCH --exclude=g[1-4]
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/g4_state_fetch_%j.log
set -euo pipefail
source experiments/feedback/g3_pie_env.sh   # HF_HOME under assets/g3_pie, telemetry off, no token
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
"$PIE_PY" -u src/feedback/g4_state_fetch.py --out assets/g4/state/hf/ST-HVG-Replogle \
  --prov reports/feedback/G4/provenance_state_fetch.json "$@"
echo "end=$(date -Is)"
