#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=28G
#SBATCH --time=00:30:00
set -euo pipefail

base="/mnt/S1/sdata/workbench/Virtual_Cell/gssong"
previous="$base/review_claude_20261121/position_steps23"
export PYTHONPATH="$PWD:$previous:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OMP_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$OPENBLAS_NUM_THREADS
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
exec "$python" -m src.tahoe_position.followup_b \
  --assets "$base/vcell_liver_r12_multiaxis/assets" \
  --common "$previous/common" --coordinates "$base/vcell_liver_position_audit/coordinates-v1" \
  --config "$previous/configs/r12_multiaxis/config.yaml" \
  --protocol "$PWD/FOLLOWUP_PROTOCOL.md" --prior-protocol "$previous/reports/tahoe_position/STEPS23_PROTOCOL.md" \
  --out "$PWD/result"
