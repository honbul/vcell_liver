#!/usr/bin/env bash
set -euo pipefail
snap=$1
families=(film attention gnn hybrid)
exec "$snap/experiments/nextgen_h3_v2/job.sbatch" "$snap" -m h3v2.fit r1 drug-0 "${families[${SLURM_ARRAY_TASK_ID:?}]}" 0 torch --smoke
