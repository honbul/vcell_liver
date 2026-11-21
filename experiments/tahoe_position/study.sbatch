#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONPATH="$PWD:$PWD/../vcell_liver_r13_optimization:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="$PWD/../vcell_liver_r11_chemical_probe/assets/study-env/bin/python"
exec "$python" -m src.tahoe_position.study "$1" \
  --assets "$PWD/../vcell_liver_r12_multiaxis/assets" \
  --history "$PWD/../vcell_liver_r13_optimization/runs/r13-round1/compound" \
  --coordinates "$PWD/coordinates-v1" --out "$PWD/$2" \
  --config configs/r12_multiaxis/config.yaml --protocol reports/tahoe_position/PROTOCOL.md
