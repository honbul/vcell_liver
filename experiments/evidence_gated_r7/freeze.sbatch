#!/usr/bin/env bash
#SBATCH -J r7_freeze
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 8G
#SBATCH -t 00:30:00
#SBATCH -o assets/logs/%x_%j.out
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
source experiments/evidence_gated_r7/freeze_args.sh
echo "start=$(date -Is) commit=$R7_COMMIT job=${SLURM_JOB_ID:-}"
"$TB_PY" -m src.evidence_gated_r7.freeze --models "$MODELS" --xpert "$XPERT" --contrasts "$CONTRASTS" --stress "$STRESS"
echo "end=$(date -Is)"
