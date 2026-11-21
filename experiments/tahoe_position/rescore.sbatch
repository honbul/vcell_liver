#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
set -euo pipefail
base="/mnt/S1/sdata/workbench/Virtual_Cell/gssong"
export PYTHONPATH="$PWD:$base/vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python="$base/vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
exec "$python" -m src.tahoe_position.rescore \
  --r12 "$base/vcell_liver_r12_multiaxis/assets" \
  --r13 "$base/vcell_liver_r13_optimization/runs/r13-round1/compound" "$@"
