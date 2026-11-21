#!/usr/bin/env bash
#SBATCH --job-name=r14-lung
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail

base="$(realpath "$PWD/../..")"
r11="$base/vcell_liver_r11_chemical_probe/assets"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:$PWD/..:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$r11/study-env/bin/python"
args=(
    "$1" --work "$PWD/r14-organ-lung"
    --assets "$base/vcell_liver_r12_multiaxis/assets" --r11 "$r11" --positions "$base/vcell_liver_position_audit/coordinates-v1"
    --data "$PWD/data" --protocol "$PWD/reports/r14_cross_organ/EXECUTION_PROTOCOL.md"
    --line-metadata /mnt/S1/sdata/pubdata/Tahoe/Tahoe-100M/metadata/cell_line_metadata.parquet
)
if [[ "$1" == predict ]]; then
    task="$SLURM_ARRAY_TASK_ID"
    roles=(A B)
    args+=(--fold "$((task / 2))" --role "${roles[$((task % 2))]}")
fi
exec "$python" -m src.r14_cross_organ.transfer_organ "${args[@]}"
