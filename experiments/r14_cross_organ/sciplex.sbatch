#!/usr/bin/env bash
#SBATCH --job-name=r14-sciplex
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
set -euo pipefail

base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
review="$base/review_claude_20261121"
cd "$1"
phase="$2"
output="$3"
export PYTHONPATH="$PWD:$review/followup_abcde:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"

args=(
    "$phase"
    --cache "$base/vcell_liver_r10_dose_composition/assets/cache/r10_v1"
    --c "$review/followup_abcde/C"
    --annotation /mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/drug_metadata.parquet
    --protocol "$PWD/reports/r14_cross_organ/EXECUTION_PROTOCOL.md"
    --output "$output"
)
if [[ "$phase" == predict ]]; then
    args+=(--task "${SLURM_ARRAY_TASK_ID:?Submit prediction as array 0-3}")
fi
exec "$python" -m src.r14_cross_organ.sciplex "${args[@]}"
