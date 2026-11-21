#!/usr/bin/env bash
#SBATCH --job-name=pos-followup-C
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

base=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
cd "$1"
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS

exec "$python" -m src.tahoe_position.followup_c \
  --cache "$base/vcell_liver_r10_dose_composition/assets/cache/r10_v1" \
  --h-receipt "$base/review_claude_20261121/position_steps23/sciplex3-h.WLLn5AL9/result/receipt.json" \
  --protocol "$PWD/reports/tahoe_position/FOLLOWUP_PROTOCOL.md" --protocol-sha256 "$2" --output "$3"
