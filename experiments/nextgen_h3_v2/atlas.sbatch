#!/usr/bin/env bash
set -euo pipefail
snap=$1
splits=(drug-0 drug-1 drug-2 cell-0 cell-1 cell-2 CTRPv2-0 CTRPv2-1 CTRPv2-2 GDSC2-0 GDSC2-1 GDSC2-2 PRISM20Q2-0 PRISM20Q2-1 PRISM20Q2-2)
exec "$snap/experiments/nextgen_h3_v2/job.sbatch" "$snap" -m h3v2.atlas "${splits[${SLURM_ARRAY_TASK_ID:?}]}"
